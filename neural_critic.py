import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from critic import Critic
from gradient_trainer import GradientTrainer
from pivotals import NN_DIMENSIONS, DISCOUNT_FACTOR_CRITIC, ELIGIBILITY_DECAY_CRITIC, LEARN_RATE_CRITIC

tf.enable_eager_execution()


class NeuralCritic(Critic):

    def __init__(self):
        super().__init__()
        self.model = self._create_model()
        plot_model(self.model, show_shapes=True)
        self.model.summary()
        self.gradient_trainer = GradientTrainer(self.model, self.eligibilities)
        self.eligibilities = None
        self.reset_eligibilities()
        self.targets = []
        self.features = []

    def _create_model(self):
        input_shape = NN_DIMENSIONS[0]
        layers = [Input(shape=(input_shape,))]
        for shape in NN_DIMENSIONS[1:]:
            layers.append(Dense(shape, activation="tanh")(layers[-1]))
        model = Model(inputs=layers[0], outputs=layers[-1])
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=LEARN_RATE_CRITIC), loss="mse", metrics=["mse"])
        return model

    def V(self, state):
        state = np.reshape(np.array(state), (1, -1))
        predicted = self.model.predict(state)
        return predicted.flatten()[0]

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

    def update(self, reward, state, new_state):
        s = np.asarray(state, dtype="float32")
        s = np.expand_dims(s, axis=0)
        # print("SSSS", s)
        target = reward + DISCOUNT_FACTOR_CRITIC * self.V(new_state)
        predicted = self.V(state)
        # print("TD_ERROR", self.delta)
        state = tf.convert_to_tensor(list(state), dtype=tf.float32)
        target = tf.convert_to_tensor([target], dtype=tf.float32)
        self.delta = target - predicted
        self.targets.append(target)
        self.features.append(state)

        self.fit(s, target)


    def reset_eligibilities(self):
        self.eligibilities = []
        for weight in self.model.trainable_weights:
            self.eligibilities.append(tf.zeros_like(weight))

    def fit(self, state, target):
        params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicted = self.model(state)
            loss = self.model.loss_functions[0](target, predicted)

            gradients = tape.gradient(loss, params)

        updated_gradients = self._update_gradients2(gradients)

        self.model.optimizer.apply_gradients(zip(updated_gradients, params))


    def _update_gradients(self, gradients, predicted):
        updated_gradients = []
        # print("Updating gradientes")
        # print(gradients)
        learn_rate = tf.convert_to_tensor(LEARN_RATE_CRITIC, dtype=tf.dtypes.float32)
        decay = tf.convert_to_tensor(ELIGIBILITY_DECAY_CRITIC, dtype=tf.dtypes.float32)
        discount = tf.convert_to_tensor(DISCOUNT_FACTOR_CRITIC, dtype=tf.dtypes.float32)
        for i, gradient in enumerate(gradients):
            update = tf.multiply(predicted, gradient)

            tmp = tf.multiply(self.eligibilities[i], discount, decay)
            self.eligibilities[i] = tf.add(tmp, update)
            # print("Gradient,", gradient)
            # print()
            # print("Predicted", predicted)
            # print()
            # print("Update", update)
            # # print(self.eligibilities[i])
            # print("shapes")
            # print(gradient.shape, self.eligibilities[i].shape)
            updated_gradients.append(self.eligibilities[i] * self.delta) # * self.delta * learn_rate)

        # print("Updated")
        # print(updated_gradients)
        # for gradient, e, param in zip(updated_gradients, self.eligibilities, self.model.trainable_weights):
        #     print(gradient.shape, e.shape, param.shape)
        return updated_gradients


    def _update_gradients2(self, gradients):
        updated_gradients = []
        decay = tf.convert_to_tensor(ELIGIBILITY_DECAY_CRITIC, dtype=tf.dtypes.float32)
        discount = tf.convert_to_tensor(DISCOUNT_FACTOR_CRITIC, dtype=tf.dtypes.float32)

        for i, gradient in enumerate(gradients):
            update = self.eligibilities[i] * discount * decay
            self.eligibilities[i] = update + gradient

            updated_gradients.append(self.eligibilities[i])

        return updated_gradients

