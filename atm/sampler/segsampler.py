import numpy as np
import torch
from tensordict import TensorDict
from einops import rearrange, repeat
from .sampler import Sampler
from atm.utils.flow_utils import sample_double_grid

class SegmentSampler(Sampler):
    def __init__(self, cfg, **kwargs):
        sampler_cfg = cfg.segment_sampler_cfg
        self.cfg = sampler_cfg
        self.segmentation_id = sampler_cfg.segmentation_id
        self.num_points = sampler_cfg.num_points
        self.mask_history = {}
        self.repeat = sampler_cfg.get("repeat", None)
        self.obs_raw_keys = sampler_cfg.get("obs_raw_keys", None)

    def sample_points(self, obs):
        if "agentview_segmentation" in obs:
            # for preprocessed demo or libero env
            geom_ids = obs["agentview_segmentation"][..., 1] # (B, H, W)
        elif "segmentation" in obs:
            # for maniskill env, obs['segmentation'].shape is (B, obs_horizon, h, w, v)
            geom_ids = obs["segmentation"].squeeze(1)[..., 0] # (B, H, W)
        else:
            raise ValueError("obs must contain 'agentview_segmentation' key")

        # agentview points
        agentview_points = []
        for gid, n in zip(self.segmentation_id, self.num_points):
            if isinstance(gid, int):
                mask = geom_ids == gid
            else:
                assert len(gid) == 2
                mask = (geom_ids >= gid[0]) & (geom_ids <= gid[1])
            agentview_points.append(self.sample_from_mask(mask, n, gid=gid))
        agentview_points = torch.cat(agentview_points, dim=-2) # (B, self.repeat, N, 2)
        # eye-in-hand points
        eye_in_hand_points = sample_double_grid(4, device="cpu")
        eye_in_hand_points = repeat(eye_in_hand_points, 'n d -> b n d', b=agentview_points.size(0))
        if self.repeat is not None:
            eye_in_hand_points = repeat(eye_in_hand_points, 'b n d -> b r n d', r=self.repeat)
            return torch.stack([agentview_points, eye_in_hand_points], dim=2) # (B, self.repeat, 2, N, 2)
        else:
            return torch.stack([agentview_points, eye_in_hand_points], dim=1) # (B, 2, N, 2)

    def sample_from_mask(self, mask, num_samples=16, gid=None):
        """
        mask: (b, h, w) torch.bool
        num_samples: int, number of samples to take
        return: (b, num_samples, 2), where this is the (u, v) coordinates of the sampled pixels in the mask
        """
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask_num_points = mask.sum()
        if mask_num_points == 0:
            if gid in self.mask_history:
                mask = self.mask_history[gid]
        else:
            if mask_num_points > 8:
                self.mask_history[gid] = mask
        b, h, w = mask.shape
        repeat = self.repeat if self.repeat is not None else 1
        points = []
        for i in range(b):
            mask_i = mask[i]
            indices = torch.nonzero(mask_i) # (N, 2)
            indices = indices.flip(1)
            if len(indices) == 0:
                points.append(torch.zeros((num_samples*repeat, 2)))
            else:
                points.append(indices[torch.randint(0, len(indices), (num_samples*repeat, ))])
        points = torch.stack(points, dim=0).float()
        points[..., 0] /= w
        points[..., 1] /= h
        if self.repeat is not None:
            points = rearrange(points, 'b (r n) d -> b r n d', r=self.repeat)
        return points

    def preprocess_demo(self, demo) -> TensorDict:
        agentview_segmentation = torch.from_numpy(demo[self.obs_raw_keys][()])
        return TensorDict(
            {
                "agentview_segmentation": agentview_segmentation
            },
            batch_size=agentview_segmentation.shape[0]
        )
    
    def reset(self):
        self.mask_history = {}