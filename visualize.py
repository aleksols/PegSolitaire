import random

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from pivotals import BOARD, BOARD_SIZE, EMPTY_CELLS, FRAME_DELAY

matplotlib.use("TkAgg")

board = BOARD(BOARD_SIZE, empty_indices=EMPTY_CELLS)

empty = list(EMPTY_CELLS)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

y_max = 0.5
y_min = -((board.size - 1) * 2 ** 0.5 + 0.5)
x_max = ((board.size - 1) * 2 ** 0.5) / 2 + 0.5
x_min = -((board.size - 1) * 2 ** 0.5) / 2 - 0.5


def animate(i):
    ax.clear()

    index = random.randint(1, 16)
    while index in empty and len(empty) < 16:
        index = random.randint(1, 16)
    empty.append(index)
    board.set_empty(empty)
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    # plt.axis("scaled")
    edges = board.get_neighbour_pairs()
    for edge in edges:
        x_start = edge[0].pos[0]
        y_start = edge[0].pos[1]
        x_end = edge[1].pos[0]
        y_end = edge[1].pos[1]
        plt.plot([x_start, x_end], [y_start, y_end], "k-", zorder=1)  # Draw black lines between cells
    for row in board.cells:
        for cell in row:
            # plt.Circle(cell.pos, radius=0.45, color=brown)
            ax.add_patch(
                matplotlib.patches.Circle(
                    xy=cell.pos,
                    radius=0.45,
                    facecolor=cell.color,
                    edgecolor="brown",
                    fill=True,
                    zorder=2
                )
            )
            # plt.text(x=cell.pos[0], y=cell.pos[1], s=cell.index, fontsize=12)


ani = animation.FuncAnimation(fig, animate, interval=FRAME_DELAY, blit=False)
plt.show()


