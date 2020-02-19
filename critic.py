import random


class Critic:
    def __init__(self, learn_rate, eligibility_decay, discount):
        self.learn_rate = learn_rate
        self.eligibility_decay = eligibility_decay
        self.discount = discount
        self.state_mapping = {}
        self.eligibilities = {}
        self.random_init = 0.005
        self.delta = 0
        self.td_errors = [self.delta]

    def V(self, state):
        # Return the critic's value for the given state, or initialize it if it does not exist
        if state not in self.state_mapping.keys():
            self.state_mapping[state] = random.uniform(0, self.random_init)
        return self.state_mapping[state]

    def e(self, state):
        if state not in self.eligibilities.keys():
            self.eligibilities[state] = 0
        return self.eligibilities[state]

    def reset_eligibilities(self):
        self.eligibilities.clear()

    def update_eligibility(self, state):
        self.eligibilities[state] *= self.discount * self.eligibility_decay

    def set_eligibility(self, state, value):
        self.eligibilities[state] = value

    def update_value(self, state):
        self.state_mapping[state] = self.V(state) + self.learn_rate * self.delta * self.e(state)

    def update_delta(self, reward, state, new_state):
        self.delta = reward + self.discount * self.V(new_state) - self.V(state)

    def update(self, reward, state, new_state):
        self.update_delta(reward, state, new_state)
        self.set_eligibility(state, 1)
