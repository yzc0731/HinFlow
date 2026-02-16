import torch
import os
import numpy as np

from atm.policy import *
from atm.utils.train_utils import setup_lr_scheduler, setup_optimizer
from .base import Agent


class HinFlow(Agent):
    def __init__(self, cfg, fabric):
        self.cfg = cfg
        self.fabric = fabric
        model_cls = eval(cfg.model_name)
        model = model_cls(**cfg.model_cfg)
        self.model = self.fabric.setup(model)
        self.extra_state_keys = self.model.extra_state_keys

        self.action_chunk = cfg.model_cfg.action_chunk
        if self.action_chunk is not None:
            self.ensemble_length = cfg.ensemble_length
            assert self.ensemble_length >= 1 and self.ensemble_length <= self.action_chunk, "Ensemble length must be less than or equal to action chunk length."
        self.action_history = []

    def reset(self):
        self.action_history = []
        self.model.reset()

    def act(self, obs, task_emb, extra_states, track_points, track=None):
        action, _tracks = self.model.act(obs, task_emb, extra_states, track_points, track)
        self.action_history.append(action)
        if self.action_chunk:
            actions = self.action_history[-self.ensemble_length:]
            l = len(actions)
            k = 0.01
            exp_weights = np.exp(-k * np.arange(l))
            exp_weights = exp_weights / exp_weights.sum()

            action = np.zeros_like(action[:, 0])
            for i in range(l):
                action += exp_weights[i] * actions[i][:, l - i - 1]

        return action, _tracks

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action, sample_points=None, online=False, **kwargs):
        return self.model.forward_loss(obs, track_obs, track, task_emb, extra_states, action, sample_points, online=online)
    
    def reset_optimizer(self, use_lr_scheduler=False):
        optimizer = setup_optimizer(self.cfg.optimizer_cfg, self.model)
        if use_lr_scheduler:
            self.scheduler = setup_lr_scheduler(optimizer, self.cfg.scheduler_cfg)
        else:
            self.scheduler = None
        self.optimizer = self.fabric.setup_optimizers(optimizer)

    def update(self, sample_replay_buffer, global_step):
        if global_step < self.cfg.num_seed_frames:
            return {}
        data = sample_replay_buffer()
        # pass a flag to indicate whether to online or pretrain 
        loss, info = self.forward_loss(**data, online=True)
        metrics = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    continue
                v = v.item()
            metrics[k] = v
        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_grad)
        self.optimizer.step()
        return metrics
    
    def pretrain_update(self, sample_replay_buffer, global_step):
        data = sample_replay_buffer()
        loss, info = self.forward_loss(**data, online=False)
        metrics = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    continue
                v = v.item()
            metrics[k] = v
        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_grad)
        self.optimizer.step()
        self.scheduler.step()
        return metrics

    def get_rewards(self, observations, actions, rewards, points, rec_tracks, real_tracks):
        return rewards

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.ckpt"))

    def load(self, path):
        self.model.load(os.path.join(path, "model.ckpt"))

    def train(self, training=True):
        self.training = training
        self.model.train(training)
    
    def __repr__(self):
        return "hinflow"
