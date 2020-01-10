from board import Board
from cell import Cell


class Triangle(Board):

    def __init__(self, size):
        super().__init__(size)
        self.cells = [[Cell(1)]]  # First cell
        counter = 2
        for row in range(1, self.size):
            new_cells = []
            for column in range(0, row + 1):
                x_pos = column
                y_pos = -row
                cell = Cell(counter, pos=(x_pos, y_pos))

                column_start = max(column - 1, 0)
                column_end = column + 1

                top_neighbours = self.cells[row - 1][column_start: column_end]
                left_neighbour = new_cells[-1:]
                neighbours = top_neighbours + left_neighbour

                cell.add_neighbour(*neighbours)
                new_cells.append(cell)
                counter += 1

            self.cells.append(new_cells)
        self.printable_cells = []
        for row in self.cells:
            for cell in row:
                self.printable_cells.append(cell)

if __name__ == '__main__':
    t = Triangle(8)
    for i in t.cells:
        print(i)

    for i in t.cells:
        for j in i:
            print(f"Neighbours for {j}:", j.neighbours)
#
