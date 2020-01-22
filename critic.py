import random

from pivotals import LEARN_RATE_CRITIC, ELIGIBILITY_DECAY_CRITIC, DISCOUNT_FACTOR_CRITIC


class Critic:
    def __init__(self):
        self.state_mapping = {}
        self.eligibilities = {}
        self.random_init = 0.0005
        self.delta = 0

    def V(self, state):
        if state not in self.state_mapping.keys():
            self.state_mapping[state] = random.uniform(0, self.random_init)
        return self.state_mapping[state]

    def e(self, state):
        if state not in self.eligibilities.keys():
            self.eligibilities[state] = 0
        return self.eligibilities[state]

    def update_eligibility(self, state, new_eligibility):
        self.eligibilities[state] = new_eligibility

    def update_value(self, state, new_value):
        self.state_mapping[state] = new_value

    def update_delta(self, reward, state, new_state):
        self.delta = reward + DISCOUNT_FACTOR_CRITIC * self.V(new_state) - self.V(state)
