import functools
import gym
import numpy as np

import pufferlib

from stonks_discrete.stonks import StonksEnv, N_ACTIONS

N_STACK = 10

def env_creator(name='stonks'):
    return functools.partial(make, name)

def make(name, buf=None):
    # Create environment
    env = StonksPuff()
    return pufferlib.emulation.GymnasiumPufferEnv(env)

class StonksPuff(StonksEnv):
    def __init__(self):
        super().__init__()
        self.n_stack = N_STACK
        n_obs = self.n_stocks * 2 + self.n_stocks * self.n_stack + 1  # per-stock: price, position; per-stock price history * n_stack; +1 overall cash
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([N_ACTIONS] * self.n_stocks)
        self.state = None

    def _get_obs(self, obs):
        price_hist = obs['price_history']
        # Take up to n_stack most recent prices
        price_hist = price_hist[-self.n_stack:]
        # Pad with earliest price if less than n_stack
        price_hist = np.pad(price_hist, (self.n_stack - len(price_hist), 0), mode='constant', constant_values=price_hist[0])
        obs['price_history'] = price_hist
        obs = np.array(obs.values())
        obs = obs.flatten()
        return obs

    def reset(self, seed):
        state = super().reset()
        self.state = state
        obs = self._get_obs(state)
        info = {}
        return obs, info

    def step(self, action):
        state, reward, done = super().step(self.state, action)
        self.state = state
        obs = self._get_obs(state)
        terminated = done
        truncated = False

        if not terminated and not truncated:
            info = {}

        return obs.astype(np.float32), reward, terminated, truncated, info


if __name__ == '__main__':
    env = env_creator()()
    obs, info = env.reset()
    breakpoint()
