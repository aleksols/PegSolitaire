from tqdm import tqdm

from pivotals import BOARD_SIZE, EMPTY_CELLS, BOARD, EPISODES, USE_NN


class ActorCriticAgent:
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
        self.critic.reset_eligibilities()
        state = self.environment.state
        self.actor.add_actions(state, self.environment.valid_actions)
        action = self.actor.action(state)

        self.critic.predictions.append(self.critic.V(state))
        if action is None:
            finished = True
        while not finished:
            reward, new_state, finished = self.environment.apply_action(action)
            state_action_sequence.append((state, action))
            G += reward
            self.actor.add_actions(new_state, self.environment.valid_actions)

            next_action = self.actor.action(new_state)

            self.actor.set_elgibility(state, action, 1)

            self.critic.update(reward, state, new_state)
            for s, a in state_action_sequence:
                self.critic.update_value(s)

                self.critic.update_eligibility(s)

                self.actor.update_policy(s, a, self.critic.delta)

                self.actor.update_eligibility(s, a)

            state = new_state
            action = next_action

        self.critic.td_errors.append(self.critic.delta)
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


def plot_results(num_pegs, actor, critic):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(10, 5)
    averages = []
    acc = 0
    for i, elem in enumerate(num_pegs, 1):
        acc += elem
        averages.append(acc / i)
    ax1.plot(num_pegs, label="num pegs")
    ax1.plot(averages, label="average")
    ax1.plot(actor.epsilon_sequence, label="epsilon")
    ax1.plot([0]*len(critic.td_errors), "black")
    ax2.plot(critic.td_errors, label="TD errors")
    avg = []
    acc = 0
    for i, elem in enumerate(critic.td_errors, 1):
        acc += elem
        avg.append(acc / i)
    ax2.plot(avg, label="average TD error")
    ax2.plot([0]*len(critic.td_errors), "black")
    ax1.legend()
    ax2.legend()
    plt.show()


def pretty_print_critic_values(critic):
    for state, value in critic.state_mapping.items():
        print("    ", state[:1], "     ")
        print("   ", state[1:3]), "    "
        print("  ", state[3:6], "    ", value)
        print(" ", state[6:10])
        print("", state[10:15])


def pretty_print_actor_saps(actor):
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


if __name__ == '__main__':
    from actor import Actor
    from neural_critic import NeuralCritic, Critic
    import matplotlib.pyplot as plt
    import matplotlib

    actor = Actor()
    if USE_NN:
        critic = NeuralCritic()
    else:
        critic = Critic()

    agent = ActorCriticAgent(actor, critic)
    rewards, num_pegs, actions = agent.play_many(EPISODES)
    # print(critic.grad)
    matplotlib.use("TkAgg")
    print(actor.epsilon)
    plot_results(num_pegs, actor, critic)
