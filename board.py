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
