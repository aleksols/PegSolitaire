class Cell:

    def __init__(self, index, pos=(0, 0), row=0, column=0):
        self.neighbours = []
        self.index = index
        self.pos = pos
        self.row = row
        self.column = column
        self.filled = True
        self.targeted = False
        self.jumped = False
        self.initial_empty = False

    def add_neighbour(self, *args):
        for cell in args:
            if cell in self.neighbours:
                continue
            self.neighbours.append(cell)
            cell.neighbours.append(self)

    @property
    def color(self):
        if self.targeted:
            return "blue"
        if self.jumped:
            return "red"
        if not self.filled and self.initial_empty:
            return "lightgreen"
        if self.filled:
            return "brown"
        return "white"

    def __hash__(self):
        return self.index

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return f"<Cell {self.index}>"


if __name__ == '__main__':
    c = Cell(0)
    c1 = Cell(1)
    print(c.__hash__())
    print(c1.__hash__())
