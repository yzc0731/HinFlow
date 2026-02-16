import json
import os
import copy
import click
import h5py
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from PIL import Image
import multiprocessing as mp
import torch.multiprocessing as mp_torch  # Add this import
from atm.utils.flow_utils import sample_from_mask, sample_double_grid
from atm.utils.cotracker_utils import Visualizer
from atm.sampler import *
from omegaconf import OmegaConf
from atm.sampler import *
from omegaconf import OmegaConf
from datetime import datetime

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

EXTRA_STATES_KEYS = [
    'gripper_states', 'joint_states', # for libero
    'tcp_pose',  # for maniskill
    'agentview_segmentation',
    'eye_in_hand_segmentation',
]

def get_device(model):
    return next(model.parameters()).device

def get_task_name_from_file_name(file_name):
    name = file_name.replace('_demo', '')
    if name[0].isupper():  # LIBERO-10 and LIBERO-90
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def track_and_remove(tracker, video, points, var_threshold=10.):
    B, T, C, H, W = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=True) # [1, T, N, 2]

    var = torch.var(pred_tracks, dim=1)  # [1, N, 2]
    var = torch.sum(var, dim=-1)[0]  # List

    # get index of points with low variance
    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        print(torch.max(var))
        assert len(idx) > 0, 'No points with low variance'

    new_points = points[:, idx].clone()

    # Repeat and sample
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, :points.shape[1]]
    # Add 10 percent height and width as noise
    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * H
    new_points[:, :, 1:] += noise

    # Track new points
    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=True)

    return pred_tracks, pred_vis


def track_through_video(video, track_model, num_points=1000):
    T, C, H, W = video.shape
    device = get_device(track_model)

    video = torch.from_numpy(video).float().to(device)

    # sample random points
    points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().to(device)
    points = torch.cat([torch.randint_like(points[:, :1], 0, T), points], dim=-1).to(device)

    # sample grid points
    grid_points = sample_double_grid(7, device=device) # (98, 2)
    grid_points[:, 0] = grid_points[:, 0] * H
    grid_points[:, 1] = grid_points[:, 1] * W
    # add timestamps
    grid_points = torch.cat([torch.randint_like(grid_points[:, :1], 0, T), grid_points], dim=-1).to(device)

    pred_tracks, pred_vis = track_and_remove(track_model, video[None], points[None])

    pred_grid_tracks, pred_grid_vis = track_and_remove(track_model, video[None], grid_points[None], var_threshold=0.)

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


@torch.no_grad()
def track_through_video_with_points(video, points, track_model, num_points=1000):
    """
    video: (t + num_track_ts, c, h, w)
    points: (t, n, 2), normalized
    """
    T, C, H, W = video.shape
    device = next(track_model.parameters()).device
    video = torch.from_numpy(video).float().to(device)
    # pad points with timestamps
    timestamps = torch.arange(points.shape[0]).unsqueeze(1).repeat(1, points.shape[1]).unsqueeze(-1) # (t, n, 1)
    # change these two tensors to the same dtype float
    timestamps = timestamps.float()
    points = points.float()
    points = torch.cat([timestamps, points], dim=-1) # (t, n, 3)
    points = points.reshape(-1, 3) # (t*n, 3)

    if points.shape[0] > num_points:
        points = points[torch.randperm(points.shape[0])[:num_points]]
    else:
        # add random points
        random_points = torch.cat([torch.randint(0, T, (num_points, 1)), torch.rand(num_points, 2)], dim=-1) # (num_points, 3)
        points = torch.cat([points, random_points], dim=0)[:num_points]
        print("replace points with random points")

    points[:, 1] *= H
    points[:, 2] *= W
    points = points.to(device)
    pred_tracks, pred_vis = track_and_remove(track_model, video[None], points[None])
    return pred_tracks.cpu(), pred_vis.cpu()

def recurrent_sample_points(demo, point_sampler):
    obs = point_sampler.preprocess_demo(demo)
    point_seq = []
    for i in range(obs.shape[0]):
        points = point_sampler.sample_points(obs[i:i+1])
        point_seq.append(points)
    point_seq = torch.cat(point_seq, dim=0)
    return point_seq

def collect_states_from_demo(h5_file, image_save_dir, demos_group, demo_k, view_names, track_model, point_sampler, task_emb, num_points, visualizer, save_vis=False, use_points=False, env_type="libero"):
    actions = np.array(demos_group[demo_k]['actions'])
    root_grp = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
    if "actions" not in root_grp:
        root_grp.create_dataset("actions", data=actions)

    if "extra_states" not in root_grp:
        extra_states_grp = root_grp.create_group("extra_states")
        for state_key in EXTRA_STATES_KEYS:
            if state_key in demos_group[demo_k]['obs']:
                extra_state = np.array(demos_group[demo_k]['obs'][state_key])
                extra_states_grp.create_dataset(state_key, data=extra_state)

    if "task_emb_bert" not in root_grp:
        if task_emb is not None:
            root_grp.create_dataset("task_emb_bert", data=task_emb)
        else:
            print("Warning: task_emb is None, skipping saving task_emb_bert.")

    for view in view_names:
        rgb = np.array(demos_group[demo_k]['obs'][f'{view}_rgb'])
        if env_type == "libero":
            rgb = rgb[:, ::-1, :, :].copy()  # The images in the raw Libero dataset is upsidedown, so we need to flip it
        rgb = rearrange(rgb, "t h w c -> t c h w")
        T, C, H, W = rgb.shape

        sample_points = recurrent_sample_points(demos_group[demo_k], point_sampler)
        if point_sampler.repeat is not None:
            sample_points = sample_points[:, 0]

        if use_points:
            if view == "agentview" and sample_points.shape[1] == 2:
                sample_points = sample_points[:, 0] # (t, n, 2)
                pred_tracks, pred_vis = track_through_video_with_points(rgb, sample_points, track_model, num_points=num_points)
            elif view == "eye_in_hand" and sample_points.shape[1] == 2:
                sample_points = sample_points[:, 1] # (t, n, 2)
                pred_tracks, pred_vis = track_through_video_with_points(rgb, sample_points, track_model, num_points=num_points)
        else:
            pred_tracks, pred_vis = track_through_video(rgb, track_model, num_points=num_points)

        if save_vis:
            visualizer.visualize(torch.from_numpy(rgb)[None], pred_tracks, pred_vis, filename=f"{demo_k}_{view}")

        # [1, T, N, 2], normalize coordinates to [0, 1] for in-picture coordinates
        pred_tracks[:, :, :, 0] /= W
        pred_tracks[:, :, :, 1] /= H

        # hierarchically save arrays under the view name
        view_grp = root_grp.create_group(view) if view not in root_grp else root_grp[view]
        if "video" not in view_grp:
            view_grp.create_dataset("video", data=rgb[None].astype(np.uint8))

        # we always update the tracks and vis when you run this script
        if "tracks" in view_grp:
            view_grp.__delitem__("tracks")
        if "vis" in view_grp:
            view_grp.__delitem__("vis")
        view_grp.create_dataset("tracks", data=pred_tracks.cpu().numpy())
        view_grp.create_dataset("vis", data=pred_vis.cpu().numpy())

        # save image pngs
        save_images(rearrange(rgb, "t c h w -> t h w c"), image_save_dir, view)


def save_images(video, image_dir, view_name):
    os.makedirs(image_dir, exist_ok=True)
    for idx, img in enumerate(video):
        Image.fromarray(img).save(os.path.join(image_dir, f"{view_name}_{idx}.png"))


def inital_save_h5(path, skip_exist):
    if os.path.exists(path) and skip_exist:
        with h5py.File(path, 'r') as f:
            if ("agentview" in f["root"]) and ("eye_in_hand" in f["root"]): return None

    elif os.path.exists(path) and not skip_exist:
        print(f"Warning: {path} already exists, will overwrite it.")
        os.remove(path)

    f = h5py.File(path, 'w')
    return f


def get_attrs_and_view_names(demo_h5):
    """ Get preproception states from h5 file object. """
    attrs = json.loads(demo_h5.attrs['env_args'])
    views = attrs['env_kwargs']['camera_names']
    views.sort()

    views = [name.replace('robot0_', '') if name.endswith("eye_in_hand") else name for name in views]
    return attrs, views


def process_demo(demo_k, target_dir, task_emb, skip_exist, source_h5_path, 
                 track_model, point_sampler, gpu_id, use_points, env_type):
    """
    Process demo_k from source_h5_path and save the processed data to target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    demos_data = h5py.File(source_h5_path, 'r')['data']

    if env_type == "libero":
        # save environment meta data
        attrs, views = get_attrs_and_view_names(demos_data)
        env_meta_path = os.path.join(target_dir, 'env_meta.json')
        if not os.path.exists(env_meta_path):
            with open(env_meta_path, 'w') as fp:
                json.dump(attrs, fp)
    elif env_type == "maniskill":
        # no environment meta data
        views = ["agentview", "eye_in_hand"]

    # setup visualization class
    video_path = os.path.join(target_dir, 'videos', demo_k)
    os.makedirs(video_path, exist_ok=True)
    visualizer = Visualizer(save_dir=video_path, pad_value=0, fps=24)

    num_points = 1000
    with torch.no_grad():
        save_path = os.path.join(target_dir, f"{demo_k}.hdf5")
        h5_file_handle = inital_save_h5(save_path, skip_exist)
        if h5_file_handle is None:
            return

        try:
            image_save_dir = os.path.join(target_dir, "images", demo_k)
            os.makedirs(image_save_dir, exist_ok=True)
            collect_states_from_demo(h5_file_handle, image_save_dir, demos_data, demo_k, views, track_model, point_sampler, task_emb, num_points, visualizer, save_vis=True, use_points=use_points, env_type=env_type)
            h5_file_handle.close()
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            print(f"current_date_time: {formatted_datetime}, {save_path} is completed.")

        except Exception as e:
            print(f"Exception {e} when cuda:{gpu_id} processing save_path = {save_path}")
            h5_file_handle.close()


def worker(task_queue, gpu_id, use_points, sampler, sampler_cfg, env_type):
    """
    Args:
        task_queue: multiprocessing.Queue
        gpu_id: int
        sampler: str 
    Worker function for multiprocessing. 
    Each worker will load the cotracker model to the specified gpu_id
    and process the tasks in the task_queue.
    """
    cotracker = torch.hub.load("./third_party/co-tracker", "cotracker2", source="local")
    cotracker = cotracker.eval().to(f"cuda:{gpu_id}")
    cfg = OmegaConf.load(sampler_cfg)
    point_sampler = eval(sampler)(cfg=cfg, device=f"cuda:{gpu_id}")
    while True:
        task = task_queue.get() # (demo_k, target_dir, task_emb, skip_exist, source_h5_path)
        if task is None:
            break
        process_demo(*task, cotracker, point_sampler=point_sampler, gpu_id=gpu_id, use_points=use_points, env_type=env_type)

@click.command()
@click.option("--source_hdf5", type=str)
@click.option("--target_dir", type=str)
@click.option("--skip_exist", type=bool, default=False)
@click.option("--use_points", type=bool, default=True)
@click.option("--sampler", type=str, default="SegmentSampler")
@click.option("--sampler_cfg", type=str)
@click.option("--env_type", type=str, default="libero")
def main(source_hdf5, target_dir, skip_exist, use_points, sampler, sampler_cfg, env_type):
    assert source_hdf5.endswith(".hdf5"), "source_hdf5 should be a hdf5 file"
    os.makedirs(target_dir, exist_ok=True)

    if env_type == "libero":
        # only use cached task embedding
        base_name = os.path.basename(source_hdf5).split('.')[0]
        task_name = get_task_name_from_file_name(base_name)
        print(f"Processing {task_name} from {source_hdf5}")
        task_name_to_emb = np.load("libero/task_embedding_caches/task_emb_bert.npy", allow_pickle=True).item()
        task_emb = task_name_to_emb[task_name]
    elif env_type == "maniskill":
        # no cached task embedding
        task_emb = None

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()

    demos = h5py.File(source_hdf5, 'r')['data']
    demo_keys = natsorted(list(demos.keys()))
    print(f"Found {len(demo_keys)} demos in {source_hdf5}")
    
    for demo_k in demo_keys:
        task_queue.put((demo_k, target_dir, task_emb, skip_exist, source_hdf5))

    processes = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(target=worker, args=(task_queue, gpu_id, use_points, sampler, sampler_cfg, env_type))
        p.start()
        processes.append(p)

    for _ in range(num_gpus):
        task_queue.put(None)  # Add sentinel values to signal the end of tasks

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()