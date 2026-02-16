from .sampler import Sampler
from atm.utils.flow_utils import sample_double_grid

class GridSampler(Sampler):
    def sample_points(self, obs):
        return sample_double_grid(4, device="cpu")[None]
