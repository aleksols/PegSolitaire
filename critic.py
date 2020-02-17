import random

from pivotals import LEARN_RATE_CRITIC, ELIGIBILITY_DECAY_CRITIC, DISCOUNT_FACTOR_CRITIC


class Critic:
    def __init__(self):
        self.state_mapping = {}
        self.eligibilities = {}
        self.random_init = 0.005
        self.delta = 0
        self.td_errors = [self.delta]
        self.predictions = []

    def V(self, state):
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
        self.eligibilities[state] *= DISCOUNT_FACTOR_CRITIC * ELIGIBILITY_DECAY_CRITIC

    def set_eligibility(self, state, value):
        self.eligibilities[state] = value

    def update_value(self, state):
        self.state_mapping[state] = self.V(state) + LEARN_RATE_CRITIC * self.delta * self.e(state)

    def update_delta(self, reward, state, new_state):
        self.delta = reward + DISCOUNT_FACTOR_CRITIC * self.V(new_state) - self.V(state)

    def update(self, reward, state, new_state):
        self.update_delta(reward, state, new_state)
        self.set_eligibility(state, 1)
