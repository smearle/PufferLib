from stonks_discrete.stonks import StonksEnv
from pufferlib.environments.stonks.environment import StonksPuff

env = StonksEnv()
obs = env.reset()
print(obs)
breakpoint()
print(env.action_space)