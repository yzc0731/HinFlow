from einops import rearrange
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torchvision import transforms
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from atm.utils.env_utils import recursive_to_tensor
from atm.dataloader.utils import ImgTrackColorJitter, ImgViewDiffTranslationAug

class Buffer():
    allowed_keys = ["obs", "action", "reward", "track", "pred_track", "extra_states", "episode"]

    def __init__(self, cfg, fabric):
        self.cfg = cfg
        self.fabric = fabric
        self._episode = 0
        self.slice_len = cfg.frame_stack
        if cfg.use_continous_track:
            self.slice_len += cfg.track_len - 1
        self._buffer = self._make_replay_buffer()
        self.augment_track = cfg.augment_track
        self.aug_prob = cfg.aug_prob
        self.track_repeat = cfg.get("track_repeat", None)
        
        img_size = self.cfg.img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def _make_replay_buffer(self):
        return ReplayBuffer(
            storage=LazyTensorStorage(self.cfg.max_buffer_size, device=torch.device("cpu")),
            sampler=SliceSampler(slice_len=self.slice_len),
            pin_memory=True,
            # prefetch=4,
            batch_size=self.cfg.batch_size * self.slice_len,
            # shared=False,
        )

    def _preprocess_trajectory(self, td):
        """
        Input:
            td: dict[str: np.ndarray]
        Output:
            TensorDict
        """
        td["episode"] = torch.tensor(self._episode, dtype=torch.int).expand(td["obs"].shape[0])
        allowed_keys_td = {k: v for k, v in td.items() if k in self.allowed_keys}
        data = recursive_to_tensor(allowed_keys_td)
        if "pred_track" in data:
            data["pred_track"] = data["pred_track"][:, :, :self.cfg.track_len] # (t, v, track_len, n, 2)
        action_chunk = self.cfg.action_chunk
        if action_chunk is not None:
            action = data["action"] # (t, a)
            t = action.shape[0]
            padded_action = torch.concat([action, action[-1:].repeat_interleave(action_chunk, dim=0)], dim=0) # (t + action_chunk, a)
            stack_actions = []
            for i in range(action_chunk):
                stack_actions.append(padded_action[i:i+t])
            stack_actions = torch.stack(stack_actions, dim=1) # (t, action_chunk, a)
            data["action"] = stack_actions
        if data["obs"].dtype != torch.float32:
            data["obs"] = data["obs"].float()
        return TensorDict(data, batch_size=[td["obs"].shape[0]]) # type: ignore

    def _preprocess_sample(self, data):
        data = data.reshape(-1, self.slice_len)
        data = self.fabric.to_device(data)
        obs = data["obs"].transpose(1, 2).float() # b, v, t, c, h, w
        track = data["track"]
        if self.track_repeat is not None:
            # b, t, self.track_repeat, v, tl, n, 2
            B, T, _, V, _, N, _ = track.shape
            assert track.shape[2] == self.track_repeat
            track = rearrange(track, "b t r v tl n d -> (b t v n) r tl d")
            b0 = track.shape[0]
            track = track[torch.arange(b0), torch.randint(0, self.track_repeat, (b0,))]
            track = rearrange(track, "(b t v n) tl d -> b v t tl n d", b=B, v=V, t=T, n=N)
        else:
            track = track.transpose(1, 2) # b, v, t, track_len, n, 2
        pred_track = data["pred_track"].transpose(1, 2) if "pred_track" in data else None
        b = obs.shape[0]

        if np.random.rand() < self.aug_prob:
            obs = rearrange(obs, "b v t c h w -> (b v) t c h w")
            track = rearrange(track, "b v t tl n c -> (b v) t tl n c")
            pred_track = rearrange(pred_track, "b v t tl n c -> (b v) t tl n c") if pred_track is not None else None
            if pred_track is not None:
                NN = track.shape[-2]
                if pred_track.shape[2] > track.shape[2]:
                    pred_track = pred_track[:, :, :track.shape[2]]
                elif pred_track.shape[2] < track.shape[2]:
                    track = track[:, :, :pred_track.shape[2]]
                track = torch.cat([track, pred_track], dim=-2)
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.
            obs = rearrange(obs, "(b v) t c h w -> b v t c h w", b=b)
            if pred_track is not None:
                pred_track = track[..., NN:, :]
                track = track[..., :NN, :]
            track = rearrange(track, "(b v) t tl n c -> b v t tl n c", b=b)
            if pred_track is not None:
                pred_track = rearrange(pred_track, "(b v) t tl n c -> b v t tl n c", b=b)
                assert pred_track.shape == track.shape, f"pred_track shape {pred_track.shape} != track shape {track.shape}"

        data_fs = data
        if self.cfg.use_continous_track:
            track_len = torch.randint(self.cfg.track_len - 1, [b])
            track_len = self.fabric.to_device(track_len)
            index = track_len.unsqueeze(-1) + torch.arange(self.cfg.frame_stack+1, device=track_len.device)
            batch_idx = torch.arange(b, device=track_len.device).unsqueeze(1)

            data_fs = data.gather(dim=1, index=index) # (b, fs+1, ...)
            obs = rearrange(obs, "b v t c h w -> b t v c h w")
            obs = obs[batch_idx, index]
            obs = rearrange(obs, "b t v c h w -> b v t c h w")

            track_index = track_len.unsqueeze(-1) + torch.arange(self.cfg.track_len, device=track_len.device)
            track = track[:, :, :self.cfg.frame_stack] # (b, v, fs, track_len, n, 2)
            track_padded = torch.cat([track, track[:, :, :, [-1]].repeat_interleave(self.cfg.track_len+1, dim=3)], dim=3)  # (b, v, t, track_len+ts, n, 2)
            track_padded = track_padded.transpose(1, 3) # (b, track_len+ts, t, v, n, 2)
            track = track_padded[batch_idx, track_index]  # (b, track_len, t, v, n, 2)
            track = track.transpose(1, 3) # (b, v, t, track_len, n, 2)
            
            if pred_track is not None:
                pred_track = pred_track[:, :, :self.cfg.frame_stack]
                pred_track_padded = torch.cat([pred_track, pred_track[:, :, :, [-1]].repeat_interleave(self.cfg.track_len+1, dim=3)], dim=3)  # (b, v, t, track_len+ts, n, 2)
                pred_track_padded = pred_track_padded.transpose(1, 3) # (b, track_len+ts, t, v, n, 2)
                pred_track = pred_track_padded[batch_idx, track_index]  # (b, track_len, t, v, n, 2)
                pred_track = pred_track.transpose(1, 3) # (b, v, t, track_len, n, 2)

        processed_data = {
            "obs": obs,
            "track_obs": obs.unsqueeze(3),
            "action": data_fs["action"],
            "extra_states": data_fs["extra_states"],
            "track": track,
        }
        if "pred_track" in data:
            processed_data["pred_track"] = pred_track
        if "reward" in data:
            processed_data["reward"] = data_fs["reward"]
        if self.cfg.use_continous_track:
            processed_data["track_len"] = (self.cfg.track_len - 1) - track_len # (b,) range: [1, track_len)
        return processed_data
    
    def add(self, td):
        td = self._preprocess_trajectory(td)
        self._buffer.extend(td)
        self._episode += 1

    def sample(self):
        data = self._buffer.sample()
        data = self._preprocess_sample(data)
        return data
    
    def reset(self):
        self._buffer.empty()
        self._buffer = self._make_replay_buffer()
    
    def __len__(self):
        return len(self._buffer)
    
    def __iter__(self):
        while True:
            yield self.sample()
