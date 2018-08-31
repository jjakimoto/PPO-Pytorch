import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from .core import BaseProcessor


class AtariProcessor(BaseProcessor):
    def __init__(self, frame_width=84, frame_height=84):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.last_observation = None

    def process(self, observation):
        if self.last_observation is not None:
            observation = np.maximum(observation, self.last_observation)
        observation = resize(rgb2gray(observation),
                             (self.frame_width, self.frame_height))
        return observation.reshape((1, self.frame_width, self.frame_height))