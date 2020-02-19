from tqdm import tqdm


class ActorCriticAgent:
    def __init__(self, actor, critic, board):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.environment = board

    def play_game(self):
        finished = False
        state_action_sequence = []
        self.actor.reset_eligibilities()
        self.critic.reset_eligibilities()

        state = self.environment.state  # Initialize first state
        self.actor.add_actions(state, self.environment.valid_actions)  # Add the set of legal moves to the actors table
        action = self.actor.action(state)  # Choose state

        if action is None:  # Prevent crash if the board have no legal moves
            finished = True

        while not finished:
            reward, new_state, finished = self.environment.apply_action(action)  # Apply action and receive reinforcement
            state_action_sequence.append((state, action))
            self.actor.add_actions(new_state, self.environment.valid_actions)

            next_action = self.actor.action(new_state)
            self.actor.set_elgibility(state, action, 1)
            self.critic.update(reward, state, new_state)  # Updates delta and eligibilities

            for s, a in state_action_sequence:
                self.critic.update_value(s)  # Does nothing for NeuralCritic
                self.critic.update_eligibility(s)  # Does nothing for NeuralCritic
                self.actor.update_policy(s, a, self.critic.delta)
                self.actor.update_eligibility(s, a)

            state = new_state
            action = next_action

        self.critic.td_errors.append(self.critic.delta)  # For plotting purposes
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
