import numpy as np

class GaussianNoise:
    def __init__(self, act_shape, sigma):
        self.mu = np.zeros(act_shape)
        self.sigma = sigma
        self.rng = np.random.RandomState()

    def __call__(self):
        return self.rng.normal(self.mu, self.sigma)

    def reset(self):
        pass
