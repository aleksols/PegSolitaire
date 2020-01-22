from tqdm import tqdm

from agent import Agent
from pivotals import DISCOUNT_FACTOR_CRITIC, LEARN_RATE_CRITIC, ELIGIBILITY_DECAY_CRITIC, LEARN_RATE_ACTOR, \
    DISCOUNT_FACTOR_ACTOR, ELIGIBILITY_DECAY_ACTOR, BOARD_SIZE, EMPTY_CELLS, BOARD, EPISODES


class ActorCriticAgent(Agent):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.environment = BOARD(BOARD_SIZE, EMPTY_CELLS)

    def play_game(self):
        finished = False
        state_action_sequence = []

        state = self.environment.state
        self.actor.add_actions(state, self.environment.valid_actions)
        action = self.actor.action(state)

        while not finished:
            reward, new_state, finished = self.environment.apply_action(action)
            state_action_sequence.append((state, action))

            self.actor.add_actions(new_state, self.environment.valid_actions)
            next_action = self.actor.action(new_state)

            self.actor.update_eligibility(new_state, next_action, 1)
            self.critic.update_delta(reward, state, new_state)
            self.critic.update_eligibility(state, 1)

            for s, a in state_action_sequence:
                new_value = self.critic.V(s) + LEARN_RATE_CRITIC * self.critic.delta * self.critic.e(s)
                self.critic.update_value(s, new_value)

                new_critic_eligibility = DISCOUNT_FACTOR_CRITIC * ELIGIBILITY_DECAY_CRITIC * self.critic.e(s)
                self.critic.update_eligibility(s, new_critic_eligibility)

                policy = self.actor.policy(s, a)
                update = LEARN_RATE_ACTOR * self.critic.delta * self.actor.e(s, a)
                policy_update = policy + update
                self.actor.update_policy(s, a, policy_update)

                new_actor_eligibility = DISCOUNT_FACTOR_ACTOR * ELIGIBILITY_DECAY_ACTOR * self.actor.e(s, a)
                self.actor.update_eligibility(s, a, new_actor_eligibility)

            state = new_state
            action = next_action

        self.environment.reset()
        return sum(state), state_action_sequence


    def play_many(self, num_games):
        results = []
        action_sequence = []
        for _ in tqdm(range(num_games)):
            num_pegs, sequence = self.play_game()
            self.actor.update_epsilon()
            results.append(num_pegs)
            action_sequence.append([elem[1] for elem in sequence])
        return results, action_sequence


if __name__ == '__main__':
    from actor import Actor
    from critic import Critic

    actor = Actor()
    critic = Critic()
    agent = ActorCriticAgent(actor, critic)
    s, actions = agent.play_many(EPISODES)
    for key in actor.state_mapping.keys():
        if sum(key) == 2:
            for k, value in actor.state_mapping[key].items():
                print(k, value)

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")
    # plt.plot(s)
    # plt.show()

    print(actor.epsilon)
    print(actions[-1])
    # actor.epsilon = 0
    # s = agent.play_many(100, debug=True)
    plt.plot(s)
    plt.show()