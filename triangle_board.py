from board import Board
from cell import Cell


class Triangle(Board):

    def __init__(self, size, empty_indices=None):
        super().__init__(size)
        self.cells = None
        self.init_cells()
        if empty_indices is not None:
            self.set_empty(empty_indices)

    def init_cells(self):
        self.cells = [[Cell(1)]]  # First cell
        counter = 2
        for row in range(1, self.size):
            new_cells = []
            for column in range(0, row + 1):
                x_pos = column
                y_pos = -row
                x = x_pos - row * 0.5
                y = y_pos
                cell = Cell(counter, pos=(x, y))

                column_start = max(column - 1, 0)
                column_end = column + 1

                top_neighbours = self.cells[row - 1][column_start: column_end]
                left_neighbour = new_cells[-1:]
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
    t = Triangle(8)
    for i in t.cells:
        print(i)

    for i in t.cells:
        for j in i:
            print(f"Neighbours for {j}:", j.neighbours)
#
