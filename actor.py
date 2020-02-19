import random


class Actor:

    def __init__(self, epsilon, epsilon_decay, learn_rate, discount, eligibility_decay):
        self.state_mapping = {}
        self.eligibilities = {}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learn_rate = learn_rate
        self.discount = discount
        self.eligibility_decay = eligibility_decay
        self.epsilon_sequence = [self.epsilon]

    def e(self, state, action):
        if (state, action) not in self.eligibilities.keys():
            self.eligibilities[(state, action)] = 0
        return self.eligibilities[(state, action)]

    def reset_eligibilities(self):
        self.eligibilities.clear()

    def update_eligibility(self, state, action):
        self.eligibilities[(state, action)] = self.discount * self.eligibility_decay * self.e(state, action)

    def set_elgibility(self, state, action, value):
        self.eligibilities[(state, action)] = value

    def update_policy(self, state, action, td_error):
        self.state_mapping[state][(state, action)] += self.learn_rate * td_error * self.e(state, action)

    def add_actions(self, state, args):
        if not args:
            return
        for action in args:
            if state not in self.state_mapping.keys():
                self.state_mapping[state] = {(state, action): 0}
            elif (state, action) not in self.state_mapping[state].keys():
                self.state_mapping[state][(state, action)] = 0

    def action(self, state):
        if state not in self.state_mapping.keys():
            return None

        action_func = self._choose_best
        r = random.random()
        if r < self.epsilon:  # Use epsilon greedy method to choose randomly sometimes
            action_func = self._choose_random
        return action_func(state)

    def _choose_random(self, state):
        if not self.state_mapping[state]:
            return None
        return random.choice(list(self.state_mapping[state].keys()))[1]  # Random uniform choice

    def _choose_best(self, state):
        max_action = [None]
        max_value = float("-inf")
        for state, action in self.state_mapping[state].keys():
            value = self.state_mapping[state][(state, action)]
            if value > max_value:
                max_action = [action]
                max_value = value
            elif value == max_value:
                max_action.append(action)

        # In case there are several actions with highest value, choose random
        if len(max_action) > 1:
            return random.choice(max_action)

        return max_action[0]

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon_sequence.append(self.epsilon)  # For plotting purposes
