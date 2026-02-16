import random
import h5py
import hydra
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist
from tensordict.tensordict import TensorDict
import lightning
from lightning.fabric import Fabric

import os
import wandb
import json
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from einops import rearrange
import pandas as pd

from atm.policy import *
from atm.sampler import *
from atm.agent import *
from atm.utils.train_utils import init_wandb
from atm.utils.log_utils import MetricLogger, BestAvgLoss
from atm.utils.env_utils import build_env, DataCollectionWrapper
from atm.utils.flow_utils import combine_track_and_img, sample_tracks_nearest_to_points
from atm.utils.replay_buffer_utils import Buffer
from atm.utils.video_utils import make_grid_video_from_numpy
from atm.utils.noise_utils import GaussianNoise
from engine.utils import rollout, solve_rotation_cfg, save_success_rate
import threading
# maniskill
from engine.maniskill_utils import make_eval_envs, maniskill_preprocess_obs, FlattenRGBDObservationWrapper, ManiskillDataCollectionWrapper

obs_key_mapping = {
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
    "tcp_pose": "tcp_pose"
}

class Workspace:
    def __init__(self, cfg):
        self.work_dir = HydraConfig.get().runtime.output_dir
        OmegaConf.save(config=cfg, f=os.path.join(self.work_dir, "config.yaml"))
        self.cfg = cfg
        self.train_env_cfg = {**cfg.env_cfg}
        self.train_env_cfg["vec_env_num"] = 1
        self.writer = SummaryWriter(log_dir=self.work_dir)
        self.env_type = self.train_env_cfg.get("env_type", "libero")

        import warnings
        warnings.simplefilter("ignore")
        self.seed(cfg.seed)
        if not self.cfg.dry:
            init_wandb(cfg)
        
        self._global_step = 0
        self._global_episode = 0

        self.fabric_device_idx = cfg.fabric_device_idx
        print("[INFO] Implement on cuda:", self.fabric_device_idx)
        self.fabric = Fabric(accelerator="cuda", devices=[self.fabric_device_idx])
        self.fabric.launch()

        self._setup_agent()
        self._setup_model()
        self._setup_point_sampler(cfg=cfg, device="cuda")

        self.metric_logger = MetricLogger(delimiter=" ")
        self.best_loss_logger = BestAvgLoss(window_size=5)
        self.views = self.cfg.camera_names
    
    def _setup_agent(self):
        agent_cls = eval(self.cfg.agent_cfg.agent_name)
        self.agent: Agent = agent_cls(self.cfg.agent_cfg, self.fabric)

        if self.cfg.load_ckpt:
            self.agent.load(self.cfg.load_ckpt)

    def _setup_model(self):
        cotracker_device = self.fabric_device_idx
        self.cotracker = torch.hub.load("./third_party/co-tracker", "cotracker2", source="local").to(f"cuda:{cotracker_device}")
        self.cotracker.eval()
        self.seed(self.cfg.seed) # avoid seed=0 in co-tracker, see cotracker.py
    
    def _setup_train_env(self):
        print("[INFO] Setup train env")
        cfg = self.cfg
        if self.env_type == "libero":
            env_dict = build_env(
                img_size=cfg.img_size, 
                seed=self.cfg.seed, 
                use_init_state=False, 
                **self.train_env_cfg
            )
            self.train_env = next(iter(env_dict.values()))[1]
            self.train_env = DataCollectionWrapper(self.train_env, horizon=self.train_env_cfg["horizon"])
        elif self.env_type == "maniskill":
            env_kwargs = dict(
                control_mode="pd_ee_delta_pose",
                reward_mode="sparse",
                obs_mode="rgb+segmentation",
                render_mode="all",
                sensor_configs=dict(shader_pack="default"),
                human_render_camera_configs=dict(shader_pack="default")
            )
            seed = self.cfg.seed
            other_kwargs = dict(obs_horizon=1)
            self.train_env = make_eval_envs(
                self.train_env_cfg["task_name"], 
                num_envs=1,
                sim_backend="physx_cpu",
                env_kwargs=env_kwargs,
                other_kwargs=other_kwargs,
                video_dir=None,
                wrappers=[FlattenRGBDObservationWrapper],
            )
            self.train_env = ManiskillDataCollectionWrapper(self.train_env)
        self.rec_tracks = []
        self.pred_vids = []
        self.sample_points = []

        self.replay_buffer = Buffer(cfg.replay_buffer_cfg, self.fabric)

        if cfg.save_train_video:
            os.makedirs(os.path.join(self.work_dir, "train/videos"), exist_ok=True)
        
        self.track_thread  = None
        self.replay_buffer_lock = threading.Lock()

        if self.env_type == "libero":
            self.rollout_env = build_env(
                img_size=self.cfg.img_size, 
                **self.cfg.env_cfg, 
                seed=self.cfg.seed+0x1337, 
                use_init_state=False
            )
        elif self.env_type == "maniskill":
            rollout_env = make_eval_envs(
                self.train_env_cfg["task_name"],
                num_envs=1,
                sim_backend="physx_cpu",
                env_kwargs=env_kwargs,
                other_kwargs=other_kwargs,
                video_dir=None,
                wrappers=[FlattenRGBDObservationWrapper],
            )
            self.rollout_env = {self.train_env_cfg["task_name"]: (0, rollout_env)}
        noise_sampler_class = eval(cfg.noise_sampler_name)
        self.noise_sampler = noise_sampler_class(**cfg.noise_sampler_cfg)
    
    def _setup_point_sampler(self, cfg, device):
        if cfg.sampler == "SegmentSampler":
            self.sampler = eval(cfg.sampler)(cfg=cfg)
        else:
            raise NotImplementedError(f"Sampler {cfg.sampler} not implemented")

    def _sample_replay_buffer(self):
        with self.replay_buffer_lock:
            data = self.replay_buffer.sample()
        B = data["obs"].shape[0]
        if self.task_emb is not None:
            task_emb = torch.repeat_interleave(self.task_emb, B, dim=0)
        data["task_emb"] = task_emb
        return data

    def _sample_pretrain_replay_buffer(self):
        with self.replay_buffer_lock:
            data = self.pretrain_replay_buffer.sample()
        B = data["obs"].shape[0]
        if self.task_emb is not None:
            task_emb = torch.repeat_interleave(self.task_emb, B, dim=0)
        data["task_emb"] = task_emb
        return data

    @torch.no_grad()
    def track_through_video_with_points(self, video, points, num_points=1000):
        """
        video: (t + num_track_ts, c, h, w)
        points: (t, n, 2)
        """
        T, C, H, W = video.shape
        device = next(self.cotracker.parameters()).device

        video = torch.from_numpy(video).float().to(device)

        # pad points with timestamps
        points = torch.cat([torch.arange(points.shape[0]).unsqueeze(1).repeat(1, points.shape[1]).unsqueeze(-1), points], dim=-1) # (t, n, 3)
        points = points.reshape(-1, 3) # (t*n, 3)
        if points.shape[0] > num_points:
            points = points[torch.randperm(points.shape[0])[:num_points]]
        else:
            # add random points
            random_points = torch.cat([torch.randint(0, T, (num_points, 1)), torch.rand(num_points, 2)], dim=-1)
            points = torch.cat([points, random_points], dim=0)[:num_points]
        points[:, 1] *= H
        points[:, 2] *= W
        points = points.to(device)
        pred_tracks, pred_vis = self.cotracker(video[None], queries=points[None], backward_tracking=True)
        
        pred_tracks[:, :, :, 0] /= H
        pred_tracks[:, :, :, 1] /= W
        return pred_tracks.cpu(), pred_vis.cpu()
    
    def get_tracks(self, video, points: torch.Tensor):
        """
        video: (t, v, h, w, c)
        points: (t, n, 2) or (t, v, n, 2)
        return: (t, v, track_len, n, 2)
        """
        T, V = video.shape[:2]
        video = rearrange(video, "t v h w c -> v t c h w")
        track = []
        vi = []
        part_videos = []
        if points.ndim == 3:
            points = points.unsqueeze(1).repeat(1, V, 1, 1) # (t, v, n, 2)
        for i in range(V):
            num_vid = (T // 75) + 1
            for j in range(num_vid):
                start_idx = T * j // num_vid
                end_idx = T * (j + 1) // num_vid
                part_video = video[i, start_idx:end_idx+self.cfg.num_track_ts]
                part_videos.append((part_video, i, start_idx, points[start_idx:end_idx, i]))
        for i in range(len(part_videos)):
            pred_track, pred_vis = self.track_through_video_with_points(part_videos[i][0], part_videos[i][3])
            track.append(pred_track)
            vi.append(pred_vis)


        sample_track = []
        for i in range(len(part_videos)):
            sample_track_per_time = []
            vid_t = part_videos[i][0].shape[0]
            track_i = track[i][0]
            track_i = torch.cat([track_i, torch.repeat_interleave(track_i[-1:], self.cfg.num_track_ts, dim=0)], dim=0) # (t + num_track_ts, n, 2)
            vi_i = vi[i][0]
            vi_i = torch.cat([vi_i, torch.repeat_interleave(vi_i[-1:], self.cfg.num_track_ts, dim=0)], dim=0) # (t + num_track_ts, n)
            for t in range(vid_t):
                track_i_t = track_i[t:t+self.cfg.num_track_ts]
                sample_track_per_time.append(track_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0)) # (t, track_len, n, 2)
        recovered_tracks = [torch.zeros(T, self.cfg.num_track_ts, 1000, 2) for _ in range(V)]
        for (vid, idx, start_time, _), track_i in zip(part_videos, sample_track):
            recovered_tracks[idx][start_time:start_time+track_i.shape[0]] = track_i
        track = torch.stack(recovered_tracks, dim=1) # (t, v, track_len, n, 2)

        return track

    def _reset_train_env(self):
        self.open_cnt = 9999
        self._env_init_seed = np.random.randint(0, 99999999)
        if self.env_type == "libero":
            self.train_env.seed(self._env_init_seed)
            obs = self.train_env.reset()
        elif self.env_type == "maniskill":
            obs, tmp_info = self.train_env.reset(seed=self._env_init_seed)
        return obs
    
    def _load_hdf5_file(self, path):
        def recurrent_sample_points(demo):
            obs = self.sampler.preprocess_demo(demo)
            point_seq = []
            for i in range(obs.shape[0]):
                points = self.sampler.sample_points(obs[i:i+1])
                point_seq.append(points)
            point_seq = torch.cat(point_seq, dim=0)
            return point_seq
        
        h5_file = h5py.File(path, 'r')
        points = recurrent_sample_points(h5_file["root"])

        td = {}
        td["action"] = torch.from_numpy(h5_file["root"]["actions"][()]).float()
        T = points.shape[0]
        td["extra_states"] = {k: torch.from_numpy(h5_file["root"]["extra_states"][k][()]).float() for k in h5_file["root"]["extra_states"].keys()}
        view_obs = []
        tracks = []
        for view in self.views:
            view_obs.append(torch.from_numpy(h5_file["root"][view]['video'][()]))
            tracks.append(torch.from_numpy(h5_file["root"][view]['tracks'][()]))
        td["obs"] = torch.concat(view_obs, dim=0).transpose(0, 1).float()
        tracks = torch.concat(tracks, dim=0) # (v, t, n, 2)
        tracks = torch.concat([tracks, torch.repeat_interleave(tracks[:, -1:], repeats=self.cfg.num_track_ts, dim=1)], dim=1) # (v, t+track_len, n, 2)

        track_vis = []
        for i in range(T):
            track_i = tracks[:, i:i+self.cfg.num_track_ts]
            track_vis.append(track_i)
        track_vis = torch.stack(track_vis, dim=0) # (t, v, track_len, n, 2)
        track_repeat = self.cfg.track_repeat
        if track_repeat:
            points2 = rearrange(points, "t r v n d -> t v (r n) d", r=track_repeat)
            track = sample_tracks_nearest_to_points(track_vis, points2)
            track = rearrange(track, "t v tl (r n) d -> t r v tl n d", r=track_repeat)
        else:
            track = sample_tracks_nearest_to_points(track_vis, points)
        td["track"] = track.float()
        
        return td

    def _load_replay_buffer(self, path):
        files = os.listdir(path)
        
        self.view_mean = None
        def setup_track_normalization(td):
            track = td["track"] # b v tl n 2
            dist = torch.square(track - track[:, :, [0]]).sum(dim=-1) # b v tl n
            mean_dist = dist.mean(dim=(0, 2, 3)) # v
            self.view_mean = mean_dist.clone()
            if self.agent.__repr__() == "rl_track":
                print("Setup view normalization...")
                print("view mean", self.view_mean)
                self.agent.view_norm = self.fabric.to_device(self.view_mean)

        for i, dirpath in enumerate(files):
            data_path = os.path.join(path, dirpath)
            if os.path.isdir(data_path):
                continue
            elif data_path.endswith(".hdf5"):
                td = self._load_hdf5_file(data_path)
            self.replay_buffer.add(td)
            if self.view_mean is None:
                setup_track_normalization(td)
        
        print(f"Loaded {len(self.replay_buffer)} data")

    def add_data_to_replay_buffer(self, observations, actions, rewards, dones, infos, points, pred_tracks, real_tracks, episode, step, pred_videos=None):
        if self.cfg.save_train_video:
            rec_tracks = pred_tracks
            if self.env_type == "libero":
                video = observations["image_large"] # (t, v, h, w, c)
            elif self.env_type == "maniskill":
                video = observations["image"]
            buffer_data_obs = rearrange(observations["image"], "t v h w c -> t v c h w")
            T, V = video.shape[:2]
            video = rearrange(video, "t v h w c -> (t v) c h w")
            draw_tracks = rec_tracks.reshape(T*V, *rec_tracks.shape[2:])
            video = combine_track_and_img(draw_tracks, video)
            video = rearrange(video, "(t v) c h w -> t v h w c", t=T)

            video = np.concatenate([video[:, 0], video[:, 1]], axis=2) # (t, h, 2*w, c)
            output_name = os.path.join(self.work_dir, f"train/videos/train_{episode}_{step}.mp4")
            print(video.shape, os.path.basename(output_name))
            make_grid_video_from_numpy([video], ncol=1, speedup=1, output_name=output_name)
        
        buffer_data_extra_states = {k: observations[obs_key_mapping[k]].astype(np.float32) for k in self.agent.extra_state_keys}
        data = {
            "obs": buffer_data_obs,
            "action": actions, # (t, action_dim)
            "track": real_tracks, # (t, v, track_len, n, 2)
            "pred_track": pred_tracks, # (t, v, track_len, n, 2)
            "extra_states": buffer_data_extra_states,
            "reward": rewards.reshape(-1, 1), # (t, 1)
        }

        self.replay_buffer.add(data)
    
    def track_worker(self, observations, actions, rewards, dones, infos, points, rec_tracks, episode, step, pred_videos):
        get_tracks_obs = observations["image"].astype(np.float32)
        track_repeat = self.cfg.track_repeat
        if track_repeat is not None:
            points2 = rearrange(points, "t r v n d -> t v (r n) d")
        else:
            points2 = points
        real_tracks = self.get_tracks(get_tracks_obs, points2.float()) # (t, v, track_len, N, 2)
        real_tracks = sample_tracks_nearest_to_points(real_tracks, points2)
        if track_repeat is not None:
            real_tracks = rearrange(real_tracks, "t v tl (r n) d -> t r v tl n d", r=track_repeat)
        with self.replay_buffer_lock:
            self.add_data_to_replay_buffer(observations, actions, rewards, dones, infos, points, rec_tracks, real_tracks, episode, step, pred_videos)


    def train(self):
        self._setup_train_env()
        self.agent.train()

        obs = self._reset_train_env()
        self.task_emb = obs.get("task_emb", None)
        if self.task_emb is not None:
            self.task_emb = torch.from_numpy(self.task_emb)
            self.task_emb = self.fabric.to_device(self.task_emb)
        else:
            self.task_emb = torch.zeros([1, 768])
            self.task_emb = self.fabric.to_device(self.task_emb)

        if (self.cfg.pretrain_frames > 0):
            self.agent.reset_optimizer(use_lr_scheduler=True)
            if self.cfg.load_pretrain_data_path is not None:
                self._load_replay_buffer(self.cfg.load_pretrain_data_path)
            for s in tqdm(range(self.cfg.pretrain_frames), desc="Pretraining"):
                metrics = self.agent.pretrain_update(self._sample_replay_buffer, s)
                for k, v in metrics.items():
                    self.writer.add_scalar(f"pretrain/{k}", v, s)
                if s > 0 and s % self.cfg.pretrain_eval_every_frames == 0:
                    self.evaluate(model_name=f"pretrain_{s}")
            self.evaluate(model_name="pretrain_final")

        if not self.cfg.get("keep_pretrain_buffer", False):
            self.replay_buffer.reset()

        self.agent.reset_optimizer(use_lr_scheduler=False)

        self.sampler.reset()
        self.noise_sampler.reset()
        for step in self.metric_logger.log_every(range(self.cfg.num_train_frames), 1000, ""):
            if self.train_env.finished():
                observations, actions, rewards, dones, infos = self.train_env.process_data()
                rec_tracks = torch.cat(self.rec_tracks, dim=0).cpu() # (t, v, track_len, n, 2)
                pred_videos = None
                points = torch.cat(self.sample_points, dim=0).cpu() # (t, n, 2) or (t, v, n, 2)

                if self.track_thread is not None:
                    self.track_thread.join()
                    self.track_thread = None
                self.track_thread  = threading.Thread(target=self.track_worker, args=(observations, actions, rewards, dones, infos, points, rec_tracks, self.global_episode, self.global_step, pred_videos))
                self.track_thread.start()

                self.writer.add_scalar("train/success", self.train_env.success_count >= 3, self.global_episode)

                obs = self._reset_train_env()
                self.agent.reset()
                self.sampler.reset()
                self.noise_sampler.reset()
                self.rec_tracks = []
                self.pred_vids = []
                self.sample_points = []
                self._global_episode += 1

            rgb = obs["image"]  # (1, v, h, w, c)

            task_emb = obs.get("task_emb", None) # (1, task_emb_dim)
            if self.env_type == "maniskill":
                task_emb = torch.zeros([1, 768])
            
            extra_states = {k: obs[obs_key_mapping[k]] for k in self.agent.extra_state_keys}

            track_points = self.sampler.sample_points(obs) # (1, n, 2)
            self.sample_points.append(track_points.cpu())
            if self.cfg.track_repeat is not None:
                track_points = track_points[:, 0]
            action, _tracks = self.agent.act(rgb, task_emb, extra_states, track_points=self.fabric.to_device(track_points), track=None)

            if self.cfg.gripper_explore:
                if np.random.rand() < 1/200:
                    print("gripper explore...")
                    self.open_cnt = 0
                    self.gripper_sign = np.sign(action[0, -1].item())
                if self.open_cnt < 6:
                    assert action.shape[0] == 1
                    action[0, -1] = -0.9 * self.gripper_sign
                self.open_cnt += 1

            noise = self.noise_sampler()
            action += noise

            assert hasattr(self.cfg, "disable_rotate") and hasattr(self.cfg, "rotation_cfg"), "Please set disable_rotate and rotation_cfg in your config."
            if self.cfg.disable_rotate:
                action = solve_rotation_cfg(self.cfg.rotation_cfg, action, obs["robot0_eef_quat"], env_type=self.env_type)

            action = np.clip(action, -1, 1)
            self.rec_tracks.append(_tracks[1].cpu()) # (1, v, track_len, n, 2)
            # rollout policy to get env feedback
            obs, rew, done, info = self.train_env.step(action)
            # optimize the policy
            metrics = self.agent.update(self._sample_replay_buffer, self._global_step)
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, step)

            if step % self.cfg.eval_every_frames == 0 and step > 0:
                self.evaluate()

            self._global_step += 1

        if self.track_thread is not None:
            self.track_thread.join()
            self.track_thread = None

        self.agent.save(f"{self.work_dir}/model_final")
        None if self.cfg.dry else print(f"finished training in {wandb.run.dir}") # type: ignore
        None if self.cfg.dry else wandb.finish()
        self.train_env.close()
    
    @property
    def global_step(self):
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode

    @torch.no_grad()
    def evaluate(self, model_name=None, save_model = True):
        if model_name is None:
            model_name = self._global_step
        self.agent.eval()
        self.agent.reset()

        if "rollout_env" not in self.__dict__:
            raise NotImplementedError("Please setup rollout_env in _setup_train_env")

        assert hasattr(self.cfg, "disable_rotate") and hasattr(self.cfg, "rotation_cfg"), "Please set disable_rotate and rotation_cfg in your config."
        result = rollout(
                self.rollout_env,
                self.env_type,
                self.agent,
                self.sampler,
                num_env_rollouts=self.cfg.num_env_rollouts,
                horizon=self.cfg.env_cfg.get("horizon", None), 
                return_wandb_video=False,
                render_large_image=self.cfg.env_cfg.render_large_image,
                draw_track=True,
                success_vid_first=False,
                fail_vid_first=False,
                connect_points_with_line=True,
                disable_rotate=self.cfg.disable_rotate,
                rotation_cfg=self.cfg.rotation_cfg
            )
                
        video = None
        for k in list(result.keys()):
            if k.startswith("rollout/vis_env"):
                video = result.pop(k)
        
        model_rollout_dir = os.path.join(self.work_dir, "rollout", f"{model_name}")
        os.makedirs(model_rollout_dir, exist_ok=True)
        # save the trajectory numpy array
        for k in list(result.keys()):
            if k.startswith("rollout/track_env"):
                track_np = result.pop(k)
                np.save(os.path.join(model_rollout_dir, f"{k.replace('rollout/', '')}.npy"), track_np)

        assert video is not None
        video = rearrange(video, "B t c h w -> B t h w c")
        
        video_save_dir = os.path.join(self.work_dir, f"eval/videos/{model_name}")
        os.makedirs(video_save_dir, exist_ok=True)
        for i in range(video.shape[0]):
            make_grid_video_from_numpy([video[i]], ncol=1, speedup=1,
                                    output_name=os.path.join(video_save_dir, f"{i}.mp4"))

        success_metrics = {k: v for k, v in result.items() if k.startswith("rollout/success_env")}
        success_metrics["rollout/mean_step"] = result["rollout/mean_step"]
        save_success_rate(model_name, success_metrics, os.path.join(self.work_dir, "eval/summary.csv"))
        self.agent.reset()
        os.makedirs(os.path.join(self.work_dir, f"eval/model"), exist_ok=True)
        if save_model:
            self.agent.save(os.path.join(self.work_dir, f"eval/model/{model_name}"))
        self.agent.train()

    @staticmethod
    def seed(seed=0):
        lightning.seed_everything(seed)

from hydra.core.global_hydra import GlobalHydra
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

@hydra.main(config_path="../conf/train_hinflow_policy", version_base="1.3")
def main(cfg: DictConfig):
    # Put the import here so that running on slurm does not have import error
    torch.set_float32_matmul_precision("medium")
    
    workspace = Workspace(cfg)
    print("Start training")
    workspace.train()

if __name__ == "__main__":
    main()