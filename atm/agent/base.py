import torch

class Agent:
    def __init__(self, cfg, fabric):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def act(self, obs, task_emb, extra_states, track_points, track=None):
        raise NotImplementedError

    def update(self, sample_replay_buffer, global_step):
        raise NotImplementedError
    
    def pretrain_update(self, sample_replay_buffer, global_step):
        raise NotImplementedError
    
    def reset_optimizer(self, use_lr_scheduler=False):
        raise NotImplementedError

    def get_rewards(self, observations, actions, rewards, points, rec_tracks, real_tracks):
        return rewards
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError

    def train(self, training=True):
        raise NotImplementedError
    
    def eval(self):
        self.train(training=False)

    def __repr__(self):
        raise NotImplementedError

