import hydra
import math
from glob import glob
import pandas as pd
import numpy as np

import datetime
import torch
torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import torch.distributed as dist
import lightning
from lightning.fabric import Fabric
import gc
import logging
import os

from einops import rearrange, repeat
from omegaconf import DictConfig
from atm.sampler import *
from atm.policy import *
from atm.utils.train_utils import setup_optimizer
from atm.utils.env_utils import build_env
from engine.utils import rollout, merge_results, save_success_rate
from atm.utils.video_utils import make_grid_video_from_numpy


def get_ckp_name(file_path):
    return os.path.basename(file_path).split('.ckpt')[0].split('_')[-1]


def sort_ckp_paths(file_list, reverse=False):
    required_names = ["final", "best"]

    epoch2path = []
    name2path = {}
    for path in file_list:
        name = get_ckp_name(path)
        if name.isdigit():
            # epoch number checkpoint
            epoch = int(name)
            epoch2path.append((epoch, path))
        else:
            # final / best checkpoint
            name2path[name] = path

    sorted_paths = sorted(epoch2path, key=lambda x: x[0])
    sorted_paths = [path for _, path in sorted_paths]

    if reverse:
        sorted_paths = sorted_paths[::-1]

    for name in required_names:
        if name in name2path:
            sorted_paths.append(name2path[name])

    return sorted_paths


def get_ckp_list(exp_dir, summary_path, reverse=False):
    all_ckp_path_list = glob(os.path.join(exp_dir, "*.ckpt"))
    all_ckp_path_list = sort_ckp_paths(all_ckp_path_list, reverse=reverse)

    # If there is no summary file, we need to evaluate all the checkpoints
    if not os.path.exists(summary_path):
        return all_ckp_path_list

    all_epochs = [get_ckp_name(ckp_path) for ckp_path in all_ckp_path_list]

    df = pd.read_csv(summary_path)
    evaluated_epochs = set([str(e) for e in df['epoch'].tolist()])  # set(str)

    ckp_to_eval = []
    for epoch, path in zip(all_epochs, all_ckp_path_list):
        if epoch not in evaluated_epochs:
            ckp_to_eval.append(path)

    return ckp_to_eval


def evaluate(fabric, cfg, checkpoint, video_save_dir, 
             epoch_name,
             num_env_rollouts, render_image_size, video_speedup,
             success_vid_first, fail_vid_first, connect_points_with_line):
    suite_name = cfg.env_cfg.env_name[0]
    os.makedirs(video_save_dir, exist_ok=True)
    cfg.model_cfg.load_path = checkpoint
    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    print(f"sampler type is {cfg.sampler}")
    pointsampler = eval(cfg.sampler)(cfg=cfg, device="cuda")

    cfg.optimizer_cfg.params.lr = 0.
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)

    model, optimizer = fabric.setup(model, optimizer)

    env_type = cfg.env_cfg.env_type
    # initialize the environments in each rank
    cfg.env_cfg.render_gpu_ids = cfg.env_cfg.render_gpu_ids[fabric.global_rank] if isinstance(cfg.env_cfg.render_gpu_ids, list) else cfg.env_cfg.render_gpu_ids
    env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
    env_idx_start = env_num_each_rank * fabric.global_rank
    env_idx_end = min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name))

    all_results = []
    render_large_image = True
    for env_idx in range(env_idx_start, env_idx_end):
        print(f"evaluating ckp {checkpoint} on env {env_idx} in ({env_idx_start}, {env_idx_end})")
        if suite_name == 'maniskill':
            from engine.maniskill_utils import FlattenRGBDObservationWrapper, make_eval_envs
            # begin maniskill env
            env_kwargs = dict(
                control_mode="pd_ee_delta_pose", 
                reward_mode="sparse", 
                obs_mode="rgb+segmentation",
                render_mode="all", 
                sensor_configs=dict(shader_pack="default"),
                human_render_camera_configs=dict(shader_pack="default")
            )
            other_kwargs = dict(obs_horizon=1)
            envs = make_eval_envs(
                cfg.env_cfg.task_name, 
                num_envs=1, # num_eval_envs
                sim_backend="physx_cpu", 
                env_kwargs=env_kwargs, 
                other_kwargs=other_kwargs, 
                video_dir=None,
                wrappers=[FlattenRGBDObservationWrapper],
            )
            env_dict = {cfg.env_cfg.task_name: (0, envs)}
            result = rollout(env_dict=env_dict,
                            env_type="maniskill",
                            policy=model,
                            sampler=pointsampler,
                            num_env_rollouts=num_env_rollouts,
                            horizon=None,
                            return_wandb_video=False,
                            draw_track=True,
                            success_vid_first=success_vid_first,
                            fail_vid_first=fail_vid_first,
                            render_large_image=render_large_image,
                            connect_points_with_line=connect_points_with_line,
                            disable_rotate=cfg.disable_rotate,
                            rotation_cfg=cfg.rotation_cfg,
                            use_atm_eval=True)
        else:
            rollout_horizon = cfg.env_cfg.get("horizon", None)
            env_dict = build_env(
                img_size=(render_image_size or cfg.img_size), 
                env_idx_start_end=(env_idx, env_idx+1), 
                render_large_image=render_large_image, 
                use_init_state=False,
                **cfg.env_cfg
            )
            result = rollout(env_dict=env_dict,
                            env_type="libero",
                            policy=model,
                            sampler=pointsampler,
                            num_env_rollouts=num_env_rollouts,
                            horizon=rollout_horizon,
                            return_wandb_video=False,
                            draw_track=True,
                            success_vid_first=success_vid_first,
                            fail_vid_first=fail_vid_first,
                            render_large_image=render_large_image,
                            connect_points_with_line=connect_points_with_line,
                            disable_rotate=cfg.disable_rotate,
                            rotation_cfg=cfg.rotation_cfg,
                            use_atm_eval=True)

        # save videos
        video = None
        for k in list(result.keys()):
            if k.startswith("rollout/vis_env"):
                video = result.pop(k)
        assert video is not None

        video = rearrange(video, "B t c h w -> B t h w c")
        print(video.shape)

        basename = os.path.basename(checkpoint)
        epoch_name = os.path.splitext(basename)[0]
        video_save_dir_epoch = os.path.join(video_save_dir, epoch_name)
        os.makedirs(video_save_dir_epoch, exist_ok=True)
        for i in range(video.shape[0]): # for loop over B
            output_name = os.path.join(video_save_dir_epoch, f"{i}.mp4")
            make_grid_video_from_numpy([video[i]], ncol=1, speedup=video_speedup, output_name=output_name)
        print(f"Saved video to {video_save_dir_epoch}")

        all_results.append(result)
        del env_dict
    all_results = merge_results(all_results, compute_avg=False)

    del model
    del optimizer
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()

    return all_results

from hydra.core.global_hydra import GlobalHydra
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    save_path = cfg.save_path
    num_env_rollouts = cfg.get("num_env_rollouts", 20)
    result_suffix = f"{num_env_rollouts}"
    if cfg.disable_rotate:
        result_suffix += "_disable_rotate"

    eval_result_dir = os.path.join(save_path, f"eval_results_{result_suffix}")
    os.makedirs(eval_result_dir, exist_ok=True)

    render_image_size = cfg.get("render_image_size", cfg.img_size)
    success_vid_first = cfg.get("success_vid_first", False)
    fail_vid_first = cfg.get("fail_vid_first", False)
    connect_points_with_line = cfg.get("connect_points_with_line", True)
    video_speedup = cfg.get("video_speedup", 1)

    # currently hardcode
    suite_name = cfg.env_cfg.env_name[0]

    summary_file_path = os.path.join(eval_result_dir, f"summary_{suite_name}.csv")
    ckp_paths_to_eval = get_ckp_list(save_path, summary_file_path, reverse=True)

    setup(cfg)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
    fabric.launch()

    for ckp_path in ckp_paths_to_eval:
        epoch_name = get_ckp_name(ckp_path)

        gathered_results = [{} for _ in range(fabric.world_size)]
        results = evaluate(fabric, cfg, checkpoint=ckp_path,
                           video_save_dir=os.path.join(eval_result_dir, f"video_{suite_name}"),
                           epoch_name=epoch_name,
                           render_image_size=render_image_size, video_speedup=video_speedup,
                           num_env_rollouts=num_env_rollouts,
                           success_vid_first=success_vid_first, fail_vid_first=fail_vid_first,
                           connect_points_with_line=connect_points_with_line)
        fabric.barrier()
        dist.all_gather_object(gathered_results, results)

        if fabric.is_global_zero:
            gathered_results = merge_results(gathered_results)

            success_metrics = {k: v for k, v in gathered_results.items() if k.startswith("rollout/success_env_avg")}
            success_metrics["rollout/mean_step"] = gathered_results["rollout/mean_step"]
            save_success_rate(epoch_name, success_metrics, summary_file_path)


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    pid = os.getpid()
    print(f"The process ID (PID) is: {pid}")
    main()
