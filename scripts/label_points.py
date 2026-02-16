import os, h5py, argparse
import glob
import numpy as np
import torch
from einops import rearrange
import cv2
from omegaconf import OmegaConf

from atm.sampler import *
from atm.utils.video_utils import make_grid_video_from_numpy
from atm.utils.flow_utils import draw_traj_on_images, sample_tracks_nearest_to_points

parser = argparse.ArgumentParser()
parser.add_argument('--task',
                    choices=["libero_butter",
                             "libero_book",
                             "libero_chocolate",
                             "libero_microwave",
                             "maniskill_pokecube",
                             "maniskill_pullcubetool",
                             "maniskill_placesphere"])
parser.add_argument("--mode", choices=["bc", "atm_grid", "atm_seg"])
args = parser.parse_args()

cfg_path = f'conf/train_bc_online/{args.task}.yaml'
cfg = OmegaConf.load(cfg_path)
if args.mode == "bc" or args.mode == "atm_grid":
    sampler_type = "GridSampler"
elif args.mode == "atm_seg":
    sampler_type = "SegmentSampler"
point_sampler = eval(sampler_type)(cfg)

def scale_video(video):
    frames = []
    for i in range(video.shape[0]):
        frame = cv2.resize(video[i].transpose(1, 2, 0), (512, 512))
        frames.append(frame.transpose(2, 0, 1))
    return np.stack(frames, axis=0)

def recurrent_sample_points(demo):
    obs = point_sampler.preprocess_demo(demo)
    point_seq = []
    for i in range(obs.shape[0]):
        points = point_sampler.sample_points(obs[i:i+1])
        point_seq.append(points)
    point_seq = torch.cat(point_seq, dim=0)
    return point_seq

data_path = f'data/policy_dataset/{args.task}'
demo_files = glob.glob(f'{data_path}/*.hdf5')
video_folder = os.path.join(data_path, 'video_w_point')
os.makedirs(video_folder, exist_ok=True)
for demo_file in demo_files:
    print(f"Processing {demo_file}")
    h5_file = h5py.File(demo_file, 'r')
    basename = os.path.splitext(os.path.basename(demo_file))[0]
    root_grp = h5_file["root"]
    points = recurrent_sample_points(h5_file["root"])
    print(f"Sampled points shape: {points.shape}") # (t, repeat, v, n, d)

    tracks = []
    tracks.append(h5_file["root"]["agentview"]["tracks"])
    tracks.append(h5_file["root"]["eye_in_hand"]["tracks"])
    track = np.concatenate(tracks, axis=0)

    track = torch.from_numpy(track).transpose(0, 1)
    T = track.shape[0]
    track = torch.concat([track, track[-1].repeat(17, 1, 1, 1)], dim=0)
    tracks = []
    for i in range(T):
        tracks.append(track[i:i+16])
    track = torch.stack(tracks, dim=0) # t tl v n 2
    track = rearrange(track, 't tl v n d -> t v tl n d')
    if sampler_type == "SegmentSampler":
        track = sample_tracks_nearest_to_points(track, points[:, 0])
    elif sampler_type == "GridSampler":
        track = sample_tracks_nearest_to_points(track, points)

    x_video = h5_file["root"]['agentview']['video'][()]
    y_video = h5_file["root"]['eye_in_hand']['video'][()]
    video = np.concatenate([x_video, y_video], axis=0)

    video = rearrange(video, 'v t c h w -> (v t) c h w')
    track = rearrange(track, 't v tl n d -> (v t) tl n d')
    video = scale_video(video)

    video = draw_traj_on_images(track, video)
    video = rearrange(video, '(v t) c h w -> v t h w c', v=2)
    video = np.concatenate([video[0], video[1]], axis=2)
    print(os.path.join(video_folder, f"{basename}.mp4"))
    make_grid_video_from_numpy([video], ncol=1, speedup=1, output_name=os.path.join(video_folder, f"{basename}.mp4"))
    h5_file.close()

    h5_file = h5py.File(demo_file, 'a')
    root_grp = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
    if "sample_points" in root_grp:
        root_grp.__delitem__("sample_points")
    root_grp.create_dataset("sample_points", data=points.numpy())
    h5_file.close()
