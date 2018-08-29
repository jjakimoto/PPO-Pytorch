import sys

import gym
from rltorch import Runner
from rltorch.agents import PPOAgent
from rltorch.processors import AtariProcessor


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 4
    env = gym.make('Breakout-v0').unwrapped

    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    WINDOW_LENGTH = 4
    # state_shape = env.observation_space.shape
    state_shape = (WINDOW_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)
    action_config = {'n_action': env.action_space.n, 'type': 'integer'}
    processor = AtariProcessor(FRAME_WIDTH, FRAME_HEIGHT)
    agent = PPOAgent(state_shape, action_config, processor=processor,
                     window_length=WINDOW_LENGTH, n_epochs=3,
                     lr=2.5e-4, entropy_coef=0.01, value_loss_coef=1,
                     num_frames_per_proc=128)

    runner = Runner(env, agent, num_workers=num_workers, multi=True)
    optimzeid_agent = runner.simulate(training=True, notebook=False, render_freq=0)