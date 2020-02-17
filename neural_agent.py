from actor_critic_agent import ActorCriticAgent

class NeuralAgent(ActorCriticAgent):

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

            self.critic.update(reward, state, new_state)
            # self.critic.set_eligibility(state, 1)

            for s, a in state_action_sequence:
                self.actor.update_policy(s, a, self.critic.delta)
                self.actor.update_eligibility(s, a)

            state = new_state
            action = next_action
        self.critic.td_errors.append(self.critic.delta)
        self.environment.reset()
        return G, sum(state), state_action_sequence


if __name__ == '__main__':
    from actor import Actor
    from neural_critic import NeuralCritic, Critic
    import matplotlib.pyplot as plt
    import matplotlib
    from pivotals import USE_NN, EPISODES
    from actor_critic_agent import plot_results

    actor = Actor()
    if USE_NN:
        critic = NeuralCritic()
    else:
        critic = Critic()

    agent = NeuralAgent(actor, critic)
    rewards, num_pegs, actions = agent.play_many(EPISODES)

    matplotlib.use("TkAgg")
    print(actor.epsilon)
    # plot_results(rewards, num_pegs, actor.epsilon_sequence)
