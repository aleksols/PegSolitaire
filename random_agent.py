import random

from agent import Agent


class RandomAgent(Agent):

    def next_action(self, state, valid_actions):
        if len(valid_actions) == 0:
            return None
        action = random.choice(valid_actions)
        print(action)
        return action

    def feedback(self):
        pass
