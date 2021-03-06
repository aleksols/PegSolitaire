from tqdm import tqdm

from agent import Agent
from pivotals import BOARD_SIZE, EMPTY_CELLS, BOARD, EPISODES


class ActorCriticAgent(Agent):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.environment = BOARD(BOARD_SIZE, EMPTY_CELLS)

    def play_game(self):
        finished = False
        G = 0
        state_action_sequence = []
        self.actor.eligibilities.clear()

        state = self.environment.state
        self.actor.add_actions(state, self.environment.valid_actions)
        action = self.actor.action(state)
        self.actor.set_elgibility(state, action, 1)
        if action is None:
            finished = True
        while not finished:
            reward, new_state, finished = self.environment.apply_action(action)
            state_action_sequence.append((state, action))
            G += reward
            self.actor.add_actions(new_state, self.environment.valid_actions)
            next_action = self.actor.action(new_state)
            self.actor.set_elgibility(new_state, next_action, 1)

            self.critic.update_delta(reward, state, new_state)
            self.critic.set_eligibility(state, 1)

            for s, a in state_action_sequence:
                self.critic.update_value(s)

                self.critic.update_eligibility(s)

                self.actor.update_policy(s, a, self.critic.delta)

                self.actor.update_eligibility(s, a)

            state = new_state
            action = next_action

        self.environment.reset()
        return G, sum(state), state_action_sequence

    def play_many(self, num_games):
        results = []
        action_sequence = []
        rewards = []

        for _ in tqdm(range(num_games)):
            reward, num_pegs, sequence = self.play_game()
            self.actor.update_epsilon()
            results.append(num_pegs)
            rewards.append(reward)
            action_sequence.append([elem[1] for elem in sequence])
        return rewards, results, action_sequence


if __name__ == '__main__':
    from actor import Actor
    from critic import Critic

    actor = Actor()
    critic = Critic()
    agent = ActorCriticAgent(actor, critic)
    rewards, num_pegs, actions = agent.play_many(EPISODES)

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")
    # plt.plot(s)
    # plt.show()

    print(actor.epsilon)
    counter = 0
    value_count = {}
    for value in actor.state_mapping.values():
        for v in value.values():
            if v in value_count.keys():
                value_count[v] += 1
            else:
                value_count[v] = 1
            counter += 1
    # print(counter)
    # print(max(value_count.keys()))
    # actor.epsilon = 0
    # s = agent.play_many(100, debug=True)
    # fi
    # plt.plot(list(value_count.keys()),list(value_count.values()))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    averages = []
    acc = 0
    for i, elem in enumerate(num_pegs, 1):
        acc += elem
        averages.append(acc / i)
    ax1.plot(num_pegs)
    ax1.plot(averages)
    ax1.plot(actor.epsilon_sequence)
    ax2.plot(rewards)
    plt.show()
    # for (cstate, cvalue), (astate, avalue) in zip(critic.state_mapping.items(), actor.state_mapping.items()):
    #     print("    ", cstate[:1], "     ")
    #     print("   ", cstate[1:3]), "    "
    #     print("  ", cstate[3:6], "    ", cvalue)
    #     print(" ", cstate[6:10])
    #     print("", cstate[10:15])
    #

    for state, mapping in actor.state_mapping.items():
        print("\nNew state")
        print(state)
        for (s, a), value in mapping.items():
            print("Action:", a)
            print("    ", state[:1], "     ")
            print("   ", state[1:3]), "    "
            print("  ", state[3:6], "    ", value)
            print(" ", state[6:10])
            print("", state[10:15])
    print("\nEligibilities")
    for state, value in critic.eligibilities.items():
        print("    ", state[:1])
        print("   ", state[1:3])
        print("  ", state[3:6], "    ", value)
        print(" ", state[6:10])
        print("", state[10:15])
    # print(actor.state_mapping)
    print(actor.eligibilities)
