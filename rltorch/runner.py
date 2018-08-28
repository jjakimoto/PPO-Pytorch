from tqdm import tqdm_notebook, tqdm
from copy import deepcopy

from .env import ParallelEnv


class Runner(object):
    """Clast to run agent at given environment parallely

    Parameters
    ----------
    env: gym.Env
        OpenAI gym environment
    agent: agent instance
    num_workers: int
        The number of environments to run parallely
    multi: bool
        If True, use parallel environment even if num_workers==1
    """
    def __init__(self, env, agent, num_workers=1, multi=False):
        self.agent = agent
        self.env0 = env
        # Parallelize environment
        if multi or num_workers > 1:
            env = [deepcopy(env) for _ in range(num_workers)]
            self.env = ParallelEnv(env)
        else:
            self.env = env

    def simulate(self, n_frames=1e6, training=True, render_freq=0,
                 notebook=False, render_all=False):
        """Run agent

        Parameters
        ----------
        n_frames: int
            The number of frames to run
        training: bool
            If True, execute in training mode
        render_freq: int, (default 0)
            If 0, not render environemnt
        notebook: bool
            If True, use iterator of tqdm for notebook

        Returns
        -------
        Optimized agent
        """
        n_frames = int(n_frames)
        if notebook:
            iteration = tqdm_notebook(range(n_frames))
        else:
            iteration = tqdm(range(n_frames))
        obs = self.env.reset()
        for step in iteration:
            action = self.agent.predict(obs, training=training)
            new_obs, reward, terminal, info = self.env.step(action)
            self.agent.observe(obs, action, reward, terminal, info,
                               training=training)
            if hasattr(self.agent, 'set_new_obs'):
                self.agent.set_new_obs(new_obs)
            self.agent.fit()
            obs = new_obs
            if render_freq > 0 and step % render_freq == 0:
                if isinstance(self.env, ParallelEnv):
                    self.env.render(render_all)
                else:
                    self.env.render()
        return self.agent
