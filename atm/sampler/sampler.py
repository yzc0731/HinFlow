from tensordict import TensorDict

class Sampler:
    def __init__(self, cfg=None, device=None):
        pass
    
    def sample_points(self, obs):
        """
        obs: {k: (b, ...)}
        return: (1, n, 2) or (b, n, 2) or (b, v, n, 2)
        """
        raise NotImplementedError
    
    def preprocess_demo(self, demo) -> TensorDict:
        """
        demo: h5py.File
        return: TensorDict
        """
        T = demo["actions"].shape[0]
        return TensorDict({}, batch_size=T)
    
    def reset(self):
        pass