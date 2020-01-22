import math

from board import Board
from cell import Cell


class Diamond(Board):

    def __init__(self, size, empty_indices):
        super().__init__(size,  empty_indices)
        self.cells = None
        self.init_cells()
        self.set_empty(empty_indices, initial=True)

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
                cell = Cell(counter, pos=(x, y), row=row, column=column)

                left_neighbour = new_cells[-1:]
                top_neighbours = []
                if row != 0:
                    top_neighbours = self.cells[row - 1][column: column + 2]

                neighbours = top_neighbours + left_neighbour

                cell.add_neighbour(*neighbours)
                new_cells.append(cell)
                counter += 1

            self.cells.append(new_cells)

    @property
    def valid_actions(self):
        actions = []
        for row in self.cells:
            for cell in row:
                if not cell.filled:
                    continue
                for neighbour in cell.neighbours:
                    if not neighbour.filled:
                        continue
                    delta_row = neighbour.row - cell.row
                    delta_column = neighbour.column - cell.column
                    if abs(delta_column + delta_row) == 2:
                        continue
                    target_row = neighbour.row + delta_row
                    target_column = neighbour.column + delta_column
                    legal_row = 0 <= target_row < self.size
                    legal_column = 0 <= target_column < self.size
                    if legal_row and legal_column:
                        target_cell = self.cells[target_row][target_column]
                        if not target_cell.filled:
                            actions.append((cell, neighbour, target_cell))
        return actions


if __name__ == '__main__':
    t = Diamond(4, empty_indices=[6])

    actions = t.valid_actions
    print(actions)
    t = Diamond(4, empty_indices=[7])

    actions = t.valid_actions
    print(actions)


