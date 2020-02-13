from tqdm import tqdm

from agent import Agent
from pivotals import BOARD_SIZE, EMPTY_CELLS, BOARD, EPISODES, USE_NN


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
        if action is None:
            finished = True
        while not finished:
            reward, new_state, finished = self.environment.apply_action(action)
            state_action_sequence.append((state, action))
            G += reward
            self.actor.add_actions(new_state, self.environment.valid_actions)

            next_action = self.actor.action(new_state)

            self.actor.set_elgibility(state, action, 1)

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


def plot_results(rewards, num_pegs, epsilons):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    averages = []
    acc = 0
    for i, elem in enumerate(num_pegs, 1):
        acc += elem
        averages.append(acc / i)
    ax1.plot(num_pegs)
    ax1.plot(averages)
    ax1.plot(epsilons)
    ax2.plot(rewards)
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

    matplotlib.use("TkAgg")
    plot_results(rewards, num_pegs, actor.epsilon_sequence)
