import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

from critic import Critic

tf.enable_eager_execution()


class NeuralCritic(Critic):

    def __init__(self, learn_rate, eligibility_decay, discount, input_shape, hidden_shapes):
        super().__init__(learn_rate, eligibility_decay, discount)
        self.model = self._create_model(input_shape, hidden_shapes)
        plot_model(self.model, show_shapes=True)
        self.model.summary()
        self.eligibilities = None
        self.reset_eligibilities()

    def _create_model(self, input_shape, hidden_shapes):
        layers = [Input(shape=(input_shape,))]
        for shape in hidden_shapes:
            layers.append(Dense(shape, activation="relu")(layers[-1]))
        out = Dense(1)(layers[-1])
        model = Model(inputs=layers[0], outputs=out)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learn_rate), loss="mse", metrics=["mse"])
        return model

    def reset_eligibilities(self):
        self.eligibilities = []
        for weight in self.model.trainable_weights:
            self.eligibilities.append(tf.zeros_like(weight))

    def V(self, state):
        state = np.reshape(np.array(state), (1, -1))
        predicted = self.model.predict(state)
        return predicted.flatten()[0]

    def update(self, reward, state, new_state):
        target = reward + self.discount * self.V(new_state)
        predicted = self.V(state)

        # Convert state and target into tensors
        state = tf.convert_to_tensor([list(state)], dtype=tf.float32)
        target = tf.convert_to_tensor([target], dtype=tf.float32)

        self.delta = target - predicted
        self.fit(state, target)

    def fit(self, state, target):
        params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicted = self.model(state)
            loss = self.model.loss_functions[0](target, predicted)
            gradients = tape.gradient(loss, params)  # Note that these gradients are dL/dw and therefore contains delta
        updated_gradients = self._update_gradients(gradients)

        self.model.optimizer.apply_gradients(zip(updated_gradients, params))

    def _update_gradients(self, gradients):
        updated_gradients = []
        decay = tf.convert_to_tensor(self.eligibility_decay, dtype=tf.dtypes.float32)
        discount = tf.convert_to_tensor(self.discount, dtype=tf.dtypes.float32)

        for i, gradient in enumerate(gradients):
            update = self.eligibilities[i] * discount * decay
            self.eligibilities[i] = update + gradient
            updated_gradients.append(self.eligibilities[i])

        return updated_gradients

    def fit_v2(self, state, target):
        params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicted = self.model(state)
            loss = self.model.loss_functions[0](target, predicted)
            gradients = tape.gradient(predicted, params)
        updated_gradients = self._update_gradients(gradients)
        self.modify_weights()

    def modify_weights(self):
        updated_weights = []
        for i, weights in enumerate(self.model.get_weights()):
            updated_weights.append(tf.add(weights, self.delta * self.eligibilities[i] * self.learn_rate))

        self.model.set_weights(updated_weights)

    def update_value(self, state):
        pass

    def set_eligibility(self, state, value):
        pass

    def update_eligibility(self, state):
        pass

    def print_weights(self):
        print(self.model.layers)
        print(self.model.trainable_weights)
        for i, layer in enumerate(self.model.layers[1:]):
            weights, biases = layer.get_weights()[:2]
            weights[0, 0] = 10
            c = 0
            for n, bias in enumerate(biases):
                print(f"Bias {i} --> Layer {i + 1}, to neuron {n}: {bias}")
                c += 1
                print(c)
            c = 0
            for neuron_index, neuron in enumerate(weights):
                for weight_index, w2 in enumerate(neuron):
                    print(f"Layer {i}, neuron {neuron_index} --> Layer {i + 1}, neuron, {weight_index}: {w2}")
                    c += 1
                    print(c)

        print(self.model.trainable_weights)
