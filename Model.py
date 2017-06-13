from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
import numpy as np

class Model:
    def __init__(self, gamma, actions_count, states_count, hidden_nodes_count):
        self.gamma = gamma
        self.states_count = states_count
        self.actions_count = actions_count
        self.hidden_nodes_number = hidden_nodes_count

        self.states = 1

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))

        self.exp_replay = ExperienceReplay(self.actions_count, self.states_count, max_memory=max_memory, discount=self.gamma)

    def remember(self, state_before_move, action, reward, state_after_move):
        self.exp_replay.remember([np.array([state_before_move]), action, reward, np.array([state_after_move])])

    def clear_session(self):
        kerasBackend.clear_session()