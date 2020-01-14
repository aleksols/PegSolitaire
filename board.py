class Board:
    def __init__(self, size):
        self.cells = None
        self.size = size

    def get_neighbour_pairs(self):
        pairs = []
        finished = []
        for row in self.cells:
            for cell in row:
                for neighbour in cell.neighbours:
                    if neighbour in finished:
                        continue
                    pairs.append((cell, neighbour))
                finished.append(cell)
        return pairs

    def set_empty(self, indices, initial=False):
        if not isinstance(indices, list):
            indices = [indices]
        for row in self.cells:
            for cell in row:
                if cell.index in indices:
                    cell.filled = False
                    if initial:
                        cell.initial_empty = True

    def valid_actions(self):
        raise NotImplementedError

    def apply_aciton(self, action):
        action[0].filled = False
        action[0].targeted = False
        action[1].filled = False
        action[1].jumped = False

        action[2].filled = True

    def visualize_action(self, action):
        action[0].targeted = True
        action[1].jumped = True

    @property
    def state(self):
        state = []
        for row in self.cells:
            tmp = []
            for cell in row:
                tmp.append(int(cell.filled))
            state.append(tuple(tmp))
        return tuple(state)