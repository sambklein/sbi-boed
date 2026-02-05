import bmi
import numpy as np
import torch


class BMIDataset:
    def __init__(self, n_sample, seed=42, task="spiral-multinormal-sparse-3-3-2-2.0"):
        # Things needed for testing with our set up
        self.n_samples = n_sample
        # BMI task setupnlujeutvngvn

        self.task = bmi.benchmark.BENCHMARK_TASKS[task]
        self.dim = self.task.dim_x
        assert self.dim == self.task.dim_y
        samples = self.task.sample(n_sample, seed)
        self.current_sample = torch.Tensor(np.array(samples[0]))
        self.updated_sample = torch.Tensor(np.array(samples[1]))
        self.eig = self.task.mutual_information
