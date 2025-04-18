import functools
import numpy as np

import pufferlib

from nof1.simulation.env import TradingEnvironment

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config_path='../nof1-trading-sim/config/experiment_config_3.yaml', live_plot=False, render_mode='human', buf=None):
    '''nof1 env creation function'''
    from nof1.utils.config_manager import ConfigManager
    from nof1.data_ingestion.historical_data_reader import HistoricalDataReader

    config_manager = ConfigManager(config_path)
    config = config_manager.config
    config['simulation']['live_plot'] = live_plot
    data_path = config['data']['historical']['data_path']
    config['data']['historical']['data_path'] = '../nof1-trading-sim/' + data_path

    # Load and preprocess data
    data_reader = HistoricalDataReader(config_manager)
    states, prices, atrs, timestamps = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironment(config_manager.config, states = states, prices=prices, atrs=atrs, timestamps=timestamps)

    # data_reader = HistoricalDataReader(config_manager)
    # data, _ = data_reader.preprocess_data()
    
    # # Create environment
    # env = TradingEnvironmentPuff(config_manager.config, data)
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)

class TradingEnvironmentPuff(TradingEnvironment):
    def __init__(self, config, data):
        super().__init__(config, data)

    def reset(self):
        obs, info = super().reset()
        breakpoint()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated and not truncated:
            info = {}
        breakpoint()

        return obs.astype(np.float32), reward, terminated, truncated, info

