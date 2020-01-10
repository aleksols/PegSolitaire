from board import Board
from cell import Cell
import math

class Diamond(Board):

    def __init__(self, size, empty_indices=None):
        super().__init__(size)
        self.cells = None
        self.init_cells()
        if empty_indices is not None:
            self.set_empty(empty_indices)

    def init_cells(self):
        self.cells = []  # First cell
        counter = 1
        for row in range(0, self.size):
            new_cells = []
            for column in range(0, self.size):

                #  See rotation matrix on Wikipedia if ever in doubt about this calculation
                angle = -math.pi / 4
                x_pos = column
                y_pos = -row
                x = x_pos * math.cos(angle) - y_pos * math.sin(angle)
                y = x_pos * math.sin(angle) + y_pos * math.cos(angle)
                cell = Cell(counter, pos=(x, y))

                left_neighbour = new_cells[-1:]
                top_neighbours = []
                if row != 0:
                    top_neighbours = self.cells[row - 1][column: column + 2]

                neighbours = top_neighbours + left_neighbour

                cell.add_neighbour(*neighbours)
                new_cells.append(cell)
                counter += 1

            self.cells.append(new_cells)

    def set_empty(self, indices):
        if not isinstance(indices, list):
            indices = [indices]
        for row in self.cells:
            for cell in row:
                if cell.index in indices:
                    cell.filled = False


if __name__ == '__main__':
    t = Diamond(4)
    for i in t.cells:
        print(i)

    for i in t.cells:
        for j in i:
            print(f"Neighbours for {j}:", j.neighbours)

