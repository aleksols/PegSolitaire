class Player:

    def __init__(self, board, agent):
        self.next_action = None
        self.agent = agent
        self.board = board
        self._update_action()

    def _update_action(self):
        self.next_action = self.agent.next_action(self.board.state, self.board.valid_actions())

    def apply_action(self):
        self.board.apply_aciton(self.next_action)
        self.agent.feedback()
        self._update_action()
