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
