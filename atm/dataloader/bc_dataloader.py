import numpy as np
import torch

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_nearest_to_points


class BCDataset(BaseDataset):
    def __init__(self, track_obs_fs, repeat_sample_points, use_ground_truth_track, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.repeat_sample_points = repeat_sample_points
        self.use_ground_truth_track = use_ground_truth_track
        print(f"[BCDataloader] self.augment_track: {self.augment_track}, self.use_ground_truth_track: {self.use_ground_truth_track}, action_chunk: {self.action_chunk}")

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            sample_points = demo["root"]["sample_points"]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            sample_points = demo["root"]["sample_points"]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.
        
        if len(sample_points.shape) == 3:
            # for grid it will be (t, n, 2)
            sample_points = sample_points[:, None].repeat(2, axis=1) # (t, v, n, 2)
        
        points = torch.Tensor(sample_points[time_offset:time_offset + self.frame_stack]) # t repeat ...
        if self.repeat_sample_points:
            repeat = points.shape[1]
            # random select 0 to repeat-1 option like gather
            selected_indices = torch.randint(0, repeat, (points.shape[0],))
            selected_points = []
            for i in range(self.frame_stack):
                selected_points.append(points[i, selected_indices[i]])
            points = torch.stack(selected_points, dim=0)  # t v n 2

        # sample tracks
        if self.use_ground_truth_track: # atm_seg
            track = track.transpose(0, 1) # t v track_len n 2
            track = sample_tracks_nearest_to_points(track, points)
            track = track.transpose(0, 1) # v t track_len n 2
        else: # atm_grid & bc
            # track is useless, so omit the calculation process and set to zeros
            track = torch.zeros_like(track[:, :, :, :self.num_track_ids, :])  # v t track_len n 2

        points = points.transpose(0, 1) # v t n 2
        assert len(points.shape) == 4, f"points shape {points.shape} should be (v, t, n, 2)"
        assert points.shape[0] == len(self.views), f"{points.shape[0]} views"

        actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        if 'task_emb_bert' in demo["root"]:
            task_embs = demo["root"]["task_emb_bert"]
        else:
            task_embs = torch.zeros([768]) # for ManiSkill data, no task embedding
        extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
                        demo['root']['extra_states'].items()}

        return obs, track_transformer_obs, track, task_embs, actions, extra_states, demo_id, points
