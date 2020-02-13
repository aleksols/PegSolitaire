from critic import Critic
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from pivotals import NN_DIMENSIONS, DISCOUNT_FACTOR_CRITIC
from splitgd import SplitGD

class NeuralCritic(Critic):

    def __init__(self):
        super().__init__()
        self.model = self._create_model()
        print(self.model.trainable_weights)

    def _create_model(self):
        input_shape = NN_DIMENSIONS[0]
        inputs = Input(shape=(input_shape,))
        layer = Dense(NN_DIMENSIONS[1], activation="sigmoid")(inputs)
        for shape in NN_DIMENSIONS[2:]:
            layer = Dense(shape, activation="sigmoid")(layer)
        model = Model(inputs=inputs, outputs=layer)
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        return model


    def V(self, state):
        state = np.reshape(np.array(state), (1, -1))
        predicted = self.model.predict(state)
        return predicted.flatten()[0]

    def update_value(self, state):
        pass

    def update_delta(self, reward, state, new_state):
        self.delta = reward + DISCOUNT_FACTOR_CRITIC * self.V(new_state) - self.V(state)
        print(self.delta)