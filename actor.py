import math
import random

import numpy as np

from pivotals import INITIAL_EPSILON, EPSILON_DECAY_RATE, LEARN_RATE_ACTOR, DISCOUNT_FACTOR_ACTOR, \
    ELIGIBILITY_DECAY_ACTOR


class Actor:

    def __init__(self):
        self.state_mapping = {}
        self.eligibilities = {}
        self.epsilon = INITIAL_EPSILON
        self.epsilon_sequence = [self.epsilon]

    def e(self, state, action):
        if (state, action) not in self.eligibilities.keys():
            self.eligibilities[(state, action)] = 0
        return self.eligibilities[(state, action)]

    def policy(self, state, action):
        return self.state_mapping[state][(state, action)]

    def increment_eligibility(self, state, action):
        self.eligibilities[(state, action)] = self.e(state, action) + 1

    def update_eligibility(self, state, action):
        self.eligibilities[(state, action)] = DISCOUNT_FACTOR_ACTOR * ELIGIBILITY_DECAY_ACTOR * self.e(state, action)

    def set_elgibility(self, state, action, value):
        self.eligibilities[(state, action)] = value

    def update_policy(self, state, action, td_error):
        self.state_mapping[state][(state, action)] += LEARN_RATE_ACTOR * td_error * self.e(state, action)


    def _distribution(self, state):
        divider = sum([math.e**v for v in self.state_mapping[state].values()])
        actions = []
        dist = []
        for (s, action), value in self.state_mapping[state].items():
            actions.append(action)
            prob = math.e**value / divider
            dist.append(prob)
        # if sum(state) == 4:
        #     print("   ", state[:1])
        #     print("  ", state[1:3])
        #     print(" ", state[3:6], actions, dist)
        #     print("", state[6:10])
        #     print(state[10:])
        #     print()

        return actions, dist


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
        self._distribution(state)
        if not self.state_mapping[state]:
            return None
        return random.choice(list(self.state_mapping[state].keys()))[1]

    def _choose_best(self, state):
        self._distribution(state)
        max_action = [None]
        max_value = float("-inf")
        for state, action in self.state_mapping[state].keys():
            value = self.state_mapping[state][(state, action)]
            if value > max_value:
                max_action = [action]
                max_value = value
            elif value == max_value:
                max_action.append(action)

        if len(max_action) > 1:
            return random.choice(max_action)

        return max_action[0]

    def update_epsilon(self):
        self.epsilon *= EPSILON_DECAY_RATE
        self.epsilon_sequence.append(self.epsilon)

if __name__ == '__main__':
    a = Actor()
    p = a.policy(1, 1000)
    a.update_policy(1, 1000, 10)
    a.policy(1, 2000)
    a.update_policy(1, 2000, 20)

    print(a.action(1, [1000, 2000]))
    print(a.state_mapping)
