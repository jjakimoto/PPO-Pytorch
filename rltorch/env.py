
from multiprocessing import Process, Pipe
import gym


def worker(conn, env):
    """Execute cmd sent from remote process

    Parameters
    ----------
    conn: multiprocess.Connection instance
        Supposed to recieve command and data from remote process
    env: gym.Env instance
    """
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                conn.send(obs)
            elif cmd == 'render':
                env.render()
            elif cmd == 'close':
                env.close()
                conn.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print("KeyboardInterupt")
    finally:
        env.close()


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes

    Parameters
    ----------
    envs: list(gym.Env)
        The list of the same gym environment
    """

    def __init__(self, envs):
        assert len(envs) >= 1, 'No environment given'

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Only index 0 environment runs as a non-daemon process
        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            # Activate a remote worker as a daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(('reset', None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(('step', action))
        # 0 index process
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        # results = [(obs_1, obs_2, .., obs_n), ..., (info_1, info2, ..., info_n)]
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def close(self):
        for local in self.locals:
            local.send(('close', None))
            local.close()
        self.envs[0].close()

    def render(self, render_all=False):
        if render_all:
            for local in self.locals:
                local.send(('render', None))
            self.envs[0].render()
        else:
            self.envs[0].render()