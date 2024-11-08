import torch
from torchgfn import DiscretePolicyEstimator, Sampler
from gfn.environment import TetrisEnvironment  # Assuming this is set up as before

class TetrisGFlowNet:
    def __init__(self):
        self.env = TetrisEnvironment()
        self.policy_estimator = DiscretePolicyEstimator(self.env.n_actions)
        self.sampler = Sampler(self.policy_estimator)

    def generate_trajectory(self):
        trajectory = self.sampler.sample_trajectories(env=self.env, n=1)
        return trajectory
