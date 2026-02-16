import os
from typing  import List
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from PIL import Image
from einops import rearrange
from atm.utils.flow_utils import combine_track_and_img, draw_traj_on_images
from atm.utils.video_utils import video_pad_time
from engine.maniskill_utils import maniskill_preprocess_output, maniskill_preprocess_obs
import robosuite.utils.transform_utils as transform_utils


obs_key_mapping = {
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
    "tcp_pose": "tcp_pose"
}


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame

def solve_rotation_cfg(rotation_cfg, action, eef_quat, env_type="libero"):
    '''
    Adjust the action based on the rotation configuration.
    Args:
        rotation_cfg: rotation configuration, a dictionary with keys 'enable_x_rotate', 'enable_y_rotate', 'enable_z_rotate'
        action: action to be adjusted
        eef_quat: end effector quaternion. (xyzw) for LIBERO and (xyzw) for ManiSkill.
        env_type: 'libero' or 'maniskill'
    Returns:
        action: adjusted action
    '''
    assert action.shape[0] == eef_quat.shape[0], "action and eef_quat must have the same batch size"
    reset_rots = []

    if env_type == "libero":
        init_quats = np.array([1.0, 0.0, 0.0, 0.0])
        for eef_quat_i in eef_quat:
            curr_rot = transform_utils.quat2mat(eef_quat_i)
            target_rot = transform_utils.quat2mat(init_quats) # target rotation
            delta_rot = target_rot.dot(curr_rot.T)
            delta_quat = transform_utils.mat2quat(delta_rot)
            axis_angle_exp = transform_utils.quat2axisangle(delta_quat)
            reset_rot = np.clip(axis_angle_exp / 0.5, -1, 1)  # reset rotation
            reset_rots.append(reset_rot)
    elif env_type == "maniskill":
        from transforms3d.quaternions import quat2axangle, quat2mat, mat2quat
        init_quats = np.array([0, 1, 0, 0])
        for eef_quat_i in eef_quat:
            curr_rot = quat2mat(eef_quat_i)
            target_rot = quat2mat(init_quats)
            delta_rot = target_rot.dot(curr_rot.T)
            delta_quat = mat2quat(delta_rot)
            delta_axis, delta_angle = quat2axangle(delta_quat)
            delta_axis_angle = delta_axis * delta_angle * -1
            reset_rot = np.clip(delta_axis_angle, -1, 1)  # reset rotation
            reset_rots.append(reset_rot)
    reset_rots = np.array(reset_rots)
    rotation_mask = np.array([
        getattr(rotation_cfg, 'enable_x_rotate', True),
        getattr(rotation_cfg, 'enable_y_rotate', True),
        getattr(rotation_cfg, 'enable_z_rotate', True)
    ]) # by default all enabled
    disable_mask = ~rotation_mask
    action[:, 3:6][:, disable_mask] = reset_rots[:, disable_mask]
    return action

@torch.no_grad()
def rollout(env_dict,
            env_type,
            policy,
            sampler,
            num_env_rollouts,
            horizon,
            return_wandb_video,
            render_large_image,
            draw_track,
            success_vid_first,
            fail_vid_first,
            connect_points_with_line,
            disable_rotate,
            rotation_cfg,
            use_atm_eval=False):
    policy.eval()
    if use_atm_eval:
        policy.mark_forward_method("act_atm_eval")

    assert len(env_dict) == 1, "do not support parallel envs nowenv"
    env_description, (env_idx, env) = next(iter(env_dict.items()))
    print("env_description:", env_description)

    all_rewards = []
    all_succ = []
    all_horizon = []
    vid = []
    all_rec_track = []
    all_step_i = []

    for _ in tqdm(range(num_env_rollouts), desc="Eval"):
        sampler.reset()
        reward = None
        success = False
        last_info = None
        episode_frames = []
        random_seed = np.random.randint(0, 99999999)
        if env_type == "libero":
            env.seed(random_seed)
            obs = env.reset()
        elif env_type == "maniskill":
            obs, tmp_info = env.reset(seed=random_seed)
            obs = maniskill_preprocess_obs(obs)
        policy.reset()
        done = False
        achieve_max_steps = False
        step_i = 0
        rec_track_list = []

        while not done and not achieve_max_steps:
            rgb = obs["image"]  # (b, v, h, w, c)
            task_emb = obs.get("task_emb", None)
            if env_type == "maniskill":
                task_emb = torch.zeros([1, 768])
            extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
            track_points = sampler.sample_points(obs)
            if hasattr(sampler, "repeat") and sampler.repeat is not None:
                track_points = track_points[:, 0]

            track = None
            if use_atm_eval:
                a, _tracks = policy.act_atm_eval(rgb, task_emb, extra_states, track_points=track_points, track=track)
            else:
                a, _tracks = policy.act(rgb, task_emb, extra_states, track_points=track_points, track=track)
            
            if disable_rotate and rotation_cfg is not None:
                a = solve_rotation_cfg(rotation_cfg, a, obs["robot0_eef_quat"], env_type=env_type)

            output = env.step(a)
            if env_type == "maniskill":
                output = maniskill_preprocess_output(output)
            obs, r, done, info = output
            reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
            done = all(done)
            success = list(info["success"])
            if env_type == "libero":
                step_i += 1
                assert horizon is not None, "Libero rollout must have horizon"
                achieve_max_steps = (step_i >= horizon)
            elif env_type == "maniskill":
                step_i += 1
                achieve_max_steps = info['truncated'].all()

            if render_large_image:
                if env_type == "libero":
                    video_img = rearrange(obs["image_large"].copy(), "b v h w c -> b v c h w")
                elif env_type == "maniskill":
                    # upsample to large image
                    b, v, h, w, c = obs["image"].shape
                    img_rearrange = rearrange(obs["image"].copy(), "b v h w c -> (b v) c h w")
                    img_tensor = torch.from_numpy(img_rearrange)
                    video_img = F.interpolate(img_tensor, size=(384, 384), mode='bilinear', align_corners=False)
                    video_img = video_img.numpy()
                    video_img = rearrange(video_img, "(b v) c h w -> b v c h w", b=b, v=v)
            else:
                video_img = rearrange(obs["image"].copy(), "b v h w c -> b v c h w")
            b, _, c, h, w = video_img.shape

            if _tracks is not None:
                _rec_track = _tracks[1]
                if draw_track:
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], video_img[:, 1]], axis=-1)
                rec_track_list.append(_rec_track)

            else:
                frame = np.concatenate([video_img[:, 0], video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

            frame = render_done_to_boundary(frame, success)
            episode_frames.append(frame)

            last_info = info
            if done or achieve_max_steps:
                if done:
                    all_step_i.append(step_i)
                break

        episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
        vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

        all_rewards += reward
        all_horizon += [step_i + 1]
        all_succ += success
        rec_track_i_vid = np.concatenate(rec_track_list, axis=0)
        all_rec_track.append(rec_track_i_vid)

    vid = video_pad_time(vid)  # (b, t, c, h, w)
    vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
    all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
    all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

    if len(all_step_i) == 0:
        mean_step = 0
    else:
        mean_step = sum(all_step_i) / len(all_step_i)

    results = {}
    results["rollout/mean_step"] = mean_step
    results[f"rollout/return_env{env_idx}"] = np.mean(all_rewards)
    results[f"rollout/horizon_env{env_idx}"] = np.mean(all_horizon)
    results[f"rollout/success_env{env_idx}"] = np.mean(all_succ)
    if return_wandb_video:
        results[f"rollout/vis_env{env_idx}"] = wandb.Video(vid, fps=30, format="mp4", caption=env_description)
    else:
        results[f"rollout/vis_env{env_idx}"] = vid

    for i, track in enumerate(all_rec_track):
        results[f'rollout/track_env{i}'] = track

    return results


def merge_results(results: List[dict], compute_avg=True):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v

    if compute_avg:
        merged_results["rollout/return_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/return_env" in k]).flatten())
        merged_results["rollout/horizon_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/horizon_env" in k]).flatten())
        merged_results["rollout/success_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/success_env" in k]).flatten())
    return merged_results


def save_success_rate(epoch, success_metrics, summary_file_path):
    success_metrics = {k.replace("rollout/", ""): str(round(v, 3)) for k, v in success_metrics.items()}
    success_heads = list(success_metrics.keys())
    success_heads = ["epoch"] + success_heads

    success_metrics["epoch"] = str(epoch)
    df = pd.DataFrame(success_metrics, index=[0]) # once one line

    if os.path.exists(summary_file_path):
        old_summary = pd.read_csv(summary_file_path)
        df = pd.concat([df, old_summary], ignore_index=True) # merge

    df = df[success_heads] # resort
    df.to_csv(summary_file_path, index=False)