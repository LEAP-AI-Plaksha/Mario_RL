import random, datetime
from pathlib import Path
import gym
from sys import argv

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from mario_bases import Mario, MetricLogger, ResizeObservation, SkipFrame

env = gym_super_mario_bros.make('SuperMarioBros-v0',  apply_api_compatibility=True, render_mode='human')

env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)

if gym.__version__ < "0.26":
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

chkpt_path = f"checkpoints/mario_net_complexmovement_37.chkpt"

checkpoint = Path(chkpt_path)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 2000

for e in range(episodes):

    state = env.reset()

    while True:

        action = mario.act(state)

        # obs, reward, done, trunk, info = env.step(action)

        next_state, reward, done, trunc, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        # q, loss = mario.learn()
        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

        env.render()

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
