class Board:
    def __init__(self, size, empty_indices):
        self.cells = None
        self.size = size
        self.empty_indices = empty_indices

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
                    # if initial:
                    #     cell.initial_empty = True

    def reset(self):
        for row in self.cells:
            for cell in row:
                if cell.index in self.empty_indices:
                    cell.filled = False
                else:
                    cell.filled = True
                cell.targeted = False
                cell.jumped = False

    @property
    def valid_actions(self):
        raise NotImplementedError

    def apply_action(self, action):
        # if action[0] is None:
        #     print("None", action)
        #     return
        if action is None:
            print("None")
            return

        action[0].filled = False
        action[0].targeted = False
        action[1].filled = False
        action[1].jumped = False
        action[2].filled = True

        return self.reward, self.state, self.finished


    def visualize_action(self, action):
        action[0].targeted = True
        action[1].jumped = True


    @property
    def finished(self):
        return len(self.valid_actions) == 0

    @property
    def reward(self):
        if self.finished and sum(self.state) != 1:
            return -1
        elif self.finished:
            return 1
        return 0



    @property
    def state(self):
        state = []
        for row in self.cells:
            for cell in row:
                state.append(int(cell.filled))
        return tuple(state)
