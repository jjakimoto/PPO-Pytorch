from collections import deque, namedtuple
import warnings
import random
from six.moves import xrange
import itertools

import numpy as np

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience',
                        'state0, action, reward, state1, terminal1')

ACExperience = namedtuple('ACExperience',
                          'action, reward, terminal, value, log_prob, entropy')


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high
        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        r = xrange(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn(
            'Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    """Erase the oldest memory after reaching maxlen

    Parameters
    ----------
    maxlen: int
        The maximum number of memory
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        """Return element of buffer at specific index
        # Argument
            idx (int): Index wanted
        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        """Append an element to the buffer
        # Argument
            v (object): Element to append
        """
        self.data.append(v)

    def length(self):
        """Return the length of Deque
        # Argument
            None
        # Returns
            The lenght of deque element
        """
        return len(self.data)


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation
    # Argument
        observation (list): List of observation

    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    """Base class for memory

    Parameters
    ----------
    window_length: int
        The length to be used for input
    ignore_episode_boundaries: bool
        If False, terminal is used without considering the boundary
        when sampling or getting recdent state
    """
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        # We do not store the final state
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations


        Parameters
        ----------
        current_observation: array-like
            Last observation

        Returns
        -------
        A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            # Order: observation => action => (reward, terminal, info)
            if current_idx >= 0:
                current_terminal = self.recent_terminals[current_idx]
            else:
                break
            if not self.ignore_episode_boundaries and current_terminal:
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        state = np.concatenate(state, 0)
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory

        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences
        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = \
                sample_batch_indexes(self.window_length + 1, self.nb_entries,
                                     size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])
            state0 = np.concatenate(state0, 0)
            state1 = np.concatenate(state1, 0)

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(
                Experience(state0=state0, action=action, reward=reward,
                           state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(SequentialMemory, self).append(observation, action, reward,
                                             terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        Returns
        -------
        The number of observations
        """
        return len(self.rewards)

    @property
    def nb_states(self):
        """Return number of observations

        Returns
        -------
        The number of usable states
        """
        return len(self.observations) - self.window_length + 1

    def get_config(self):
        """Return configurations of SequentialMemory

        Returns
        -------
        Dict of Config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class ACMemory(SequentialMemory):
    def __init__(self, num_frames_per_proc, window_length, **kwargs):
        limit = num_frames_per_proc + window_length - 1
        super(ACMemory, self).__init__(limit, window_length=window_length, **kwargs)
        self.limit = limit
        self.num_frames_per_proc = num_frames_per_proc
        self.reset()

    def get_recent_state(self, current_observation):
        """Return list of last observations


        Parameters
        ----------
        current_observation: tuple(array-like)
            Each element corresponds to observation for each agent

        Returns
        -------
        Array of the last observations of agents
        """

        state_list = []
        n_workers = len(current_observation)
        for worker_idx in range(n_workers):
            state = self._get_recent_state(current_observation, worker_idx)
            state_list.append(state)
        return np.stack(state_list)

    def _get_recent_state(self, current_observation, worker_idx):
        state = [current_observation[worker_idx]]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            if current_idx >= 0:
                current_terminal = self.recent_terminals[current_idx][worker_idx]
            else:
                break
            if not self.ignore_episode_boundaries and current_terminal:
                break
            state.insert(0, self.recent_observations[current_idx][worker_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        state = np.concatenate(state, 0)
        return state

    def store_value_log_prob(self, value, log_prob, entropy):
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def sample(self):
        return ACExperience(action=self.actions[-self.num_frames_per_proc:],
                            reward=self.rewards[-self.num_frames_per_proc:],
                            terminal=self.terminals[-self.num_frames_per_proc:],
                            value=self.values[-self.num_frames_per_proc:],
                            log_prob=self.log_probs[-self.num_frames_per_proc:],
                            entropy=self.entropies[-self.num_frames_per_proc:])


    def reset(self):
        self.actions = list()
        self.rewards = list()
        self.terminals = list()
        self.observations = list()
        self.values = list()
        self.log_probs = list()
        self.entropies = list()

    def get_config(self):
        """Return configurations of ACMemory

        Returns
        -------
        Dict of Config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

