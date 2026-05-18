import numpy as np
import torch as th


class ScheduleSampler(object):
    def weights(self):
        raise NotImplementedError

    def sample(self, batch_size, device):
        weights = self.weights()
        prob = weights / np.sum(weights)
        indices_np = np.random.choice(len(prob), size=(batch_size,), p=prob)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(prob) * prob[indices_np])
        sample_weights = th.from_numpy(weights_np).float().to(device)
        return indices, sample_weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps], dtype=np.float64)

    def weights(self):
        return self._weights


class LossSecondMomentResampler(ScheduleSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_timesteps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, timesteps, losses):
        for timestep, loss in zip(timesteps, losses):
            if self._loss_counts[timestep] == self.history_per_term:
                self._loss_history[timestep, :-1] = self._loss_history[timestep, 1:]
                self._loss_history[timestep, -1] = loss
            else:
                self._loss_history[timestep, self._loss_counts[timestep]] = loss
                self._loss_counts[timestep] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.concatenate(
            [np.ones([num_timesteps // 2]), np.zeros([num_timesteps // 2]) + 0.5]
        )

    def weights(self):
        return self._weights


def create_named_schedule_sampler(name, num_timesteps):
    if name == 'uniform':
        return UniformSampler(num_timesteps)
    if name == 'lossaware':
        return LossSecondMomentResampler(num_timesteps)
    if name == 'fixstep':
        return FixSampler(num_timesteps)
    raise NotImplementedError(f'unknown schedule sampler: {name}')
