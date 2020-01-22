from board import Board
from cell import Cell


class Triangle(Board):

    def __init__(self, size, empty_indices=1):
        super().__init__(size,  empty_indices)
        self.cells = None
        self.init_cells()
        self.set_empty(empty_indices, initial=True)

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
                cell = Cell(counter, pos=(x, y), row=row, column=column)

                column_start = max(column - 1, 0)
                column_end = column + 1

                top_neighbours = self.cells[row - 1][column_start: column_end]
                left_neighbour = new_cells[-1:]
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

                    if abs(delta_column + delta_row) == 0:
                        continue

                    target_row = neighbour.row + delta_row
                    target_column = neighbour.column + delta_column

                    legal_row = 0 <= target_row < self.size
                    if not legal_row:
                        continue

                    legal_column = 0 <= target_column < len(self.cells[target_row])
                    if not legal_column:
                        continue

                    target_cell = self.cells[target_row][target_column]
                    if not target_cell.filled:
                        actions.append((cell, neighbour, target_cell))
        return actions


if __name__ == '__main__':
    t = Triangle(4, empty_indices=[6])
    print(t.valid_actions())

#
