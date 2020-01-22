import random

import numpy as np

from pivotals import INITIAL_EPSILON, EPSILON_DECAY_RATE


class Actor:

    def __init__(self):
        self.state_mapping = {}
        self.eligibilities = {}
        self.epsilon = INITIAL_EPSILON

    def e(self, state, action):
        if (state, action) not in self.eligibilities.keys():
            self.eligibilities[(state, action)] = 0
        return self.eligibilities[(state, action)]

    def policy(self, state, action):
        return self.state_mapping[state][(state, action)]

    def increment_eligibility(self, state, action):
        self.eligibilities[(state, action)] = self.e(state, action) + 1

    def update_eligibility(self, state, action, new_value):
        self.eligibilities[(state, action)] = new_value

    def update_policy(self, state, action, new_value):
        self.state_mapping[state][(state, action)] = new_value


    # def _distribution(self, state, action):
    #     divider = sum(self.eligibilities[state].values())
    #     return self.eligibilities[state][(state, action)] / divider
    def add_actions(self, state, args):
        if not args:
            self.state_mapping[state] = {}
        for action in args:
            if state not in self.state_mapping.keys():
                self.state_mapping[state] = {(state, action): 0}
            elif (state, action) not in self.state_mapping[state].keys():
                self.state_mapping[state][(state, action)] = 0


    def action(self, state):

        action_func = self._choose_best
        r = random.random()
        if r < self.epsilon:
            action_func = self._choose_random
        return action_func(state)

    def _choose_random(self, state):
        # TODO use distributions
        if not self.state_mapping[state]:
            return None
        return random.choice(list(self.state_mapping[state].keys()))[1]

    def _choose_best(self, state):
        max_action = None
        max_value = float("-inf")
        for state, action in self.state_mapping[state].keys():
            value = self.state_mapping[state][(state, action)]
            if value > max_value:
                max_action = action
                max_value = value

        return max_action

    def update_epsilon(self):
        self.epsilon *= EPSILON_DECAY_RATE

if __name__ == '__main__':
    a = Actor()
    p = a.policy(1, 1000)
    a.update_policy(1, 1000, 10)
    a.policy(1, 2000)
    a.update_policy(1, 2000, 20)

    print(a.action(1, [1000, 2000]))
    print(a.state_mapping)
