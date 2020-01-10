import random

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx

from diamond_board import Diamond
from triangle_board import Triangle

matplotlib.use("TkAgg")

G = nx.Graph()

empty = [5]
board = Triangle(8, empty_indices=empty)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)

y_max = 0.5
y_min = -((board.size - 1) * 2**0.5 + 0.5)
x_max = ((board.size - 1) * 2**0.5) / 2 + 0.5
x_min = -((board.size - 1) * 2**0.5) / 2 - 0.5

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
            plt.text(x=cell.pos[0], y=cell.pos[1], s=cell.index, fontsize=12)


ani = animation.FuncAnimation(fig, animate, interval=1000, blit=False)
plt.show()

# fig.canvas.draw()
# plt.title("Test")
# fig.canvas.draw()
#
# print(ax.patches)
