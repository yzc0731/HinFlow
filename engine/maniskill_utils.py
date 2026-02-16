from typing import Dict, Optional
from einops import rearrange
import numpy as np
import torch

import gymnasium as gym
import gymnasium.spaces.utils
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from transforms3d.quaternions import quat2axangle, quat2mat, mat2quat

def cpu_make_env(
    env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict(), wrappers=None
):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                info_on_video=True,
                source_type="bc",
                source_desc="bc evaluation rollout",
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver") # 多进程异步操作
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                    wrappers,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        # max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            print("record_video")
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="bc gpu",
                source_desc="bc evaluation rollout gpu",
                # max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


def parse_tcp_pose_from_flatten_state(state_flatten):
    state_tcp = state_flatten[..., -7:]
    return state_tcp


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
        sep_depth (bool): Whether to separate depth and rgb images in the observation. Default is True.

    Note that the returned observations will have a "rgb" or "depth" key depending on the rgb/depth bool flags, and will
    always have a "state" key. If sep_depth is False, rgb and depth will be merged into a single "rgbd" key.
    """

    def __init__(self, env, rgb=True, depth=False, state=True, sep_depth=True, seg=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.sep_depth = sep_depth
        self.include_state = state
        self.include_seg = seg

        # check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        if "segmentation" not in first_cam:
            self.include_seg = False
            print("seg not in first_cam")
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = []
        depth_images = []
        seg_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])
            if self.include_seg:
                seg_images.append(cam_data["segmentation"])

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)
        if len(seg_images) > 0:
            seg_images = torch.concat(seg_images, axis=-1)
        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb_images
        elif self.include_rgb and self.include_depth:
            if self.sep_depth:
                ret["rgb"] = rgb_images
                ret["depth"] = depth_images
            else:
                ret["rgbd"] = torch.concat([rgb_images, depth_images], axis=-1)
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth_images
        if self.include_seg:
            ret["segmentation"] = seg_images
        return ret

def maniskill_preprocess_obs(obs):
    # reshape rgb to match libero format
    rgb = obs.pop('rgb')
    rgb = rearrange(rgb, "b t h w (v c) -> b t v h w c", c=3)
    assert rgb.shape[1] == 1, f"by default obs_horizon=1, but got {rgb.shape[1]}"
    rgb = rgb.squeeze(1) # (b, v, h, w, c)
    obs['image'] = rgb

    # reshape state to match libero format
    state_flatten = obs["state"] # (b, t, dim)
    state_tcp = parse_tcp_pose_from_flatten_state(state_flatten) # (b, t, 7)
    state_tcp = state_tcp.squeeze(1) # (b, 7)
    obs['tcp_pose'] = state_tcp

    # get q from tcp_pose
    obs['robot0_eef_quat'] = state_tcp[:, 3:7] # (b, 4)
    return obs

def maniskill_preprocess_output(output):
    obs, rew, terminated, truncated, info = output
    obs = maniskill_preprocess_obs(obs)
    done = list(info["success"])
    info['terminated'] = terminated
    info['truncated'] = truncated
    return obs, rew, done, info

class ManiskillDataCollectionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ManiskillDataCollectionWrapper, self).__init__(env)
        self.reset()
    
    def reset(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 99999999)
        obs, tmp_info = self.env.reset(seed=seed)
        self._reset_memory()
        obs = maniskill_preprocess_obs(obs)
        self.observations.append(obs)
        return obs, tmp_info
    
    def process_data(self):
        """
        Merge the collected data into numpy arrays.
        """
        observations = self.observations
        observations.pop() # remove the last observation
        observations = {k: np.concatenate([obs[k] for obs in observations], axis=0) for k in observations[0].keys()} # {k: (t, ...)}
        actions = np.concatenate(self.actions) # (t, action_dim)
        rewards = np.concatenate(self.rewards) # (t,)
        dones = np.concatenate(self.dones_list) # (t,)
        infos = None
        # infos = self.infos_list
        # infos = {k: np.concatenate([info[k] for info in infos], axis=0) for k in infos[0].keys()} # {k: (t, ...)}
        return observations, actions, rewards, dones, infos

    def finished(self):
        # finished if we have success or reached max steps
        done = self.success_count >= 4
        return done or self.achieve_max_steps

    def _reset_memory(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones_list = []
        self.infos_list = []
        self.success_count = 0
        self.achieve_max_steps = False
    
    def step(self, action):
        output = self.env.step(action)
        obs, rew, done, info = maniskill_preprocess_output(output)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(rew)
        self.dones_list.append(done)
        self.infos_list.append(info)
        self.success_count += all(list(info["success"]))
        self.achieve_max_steps = info["truncated"].all()
        return obs, rew, done, info