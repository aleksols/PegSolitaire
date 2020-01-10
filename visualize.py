import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from triangle_board import Triangle

matplotlib.use("TkAgg")
G = nx.Graph()
T = Triangle(4)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
plt.axis('scaled')


def animate(i):
    ax.clear()
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.axis('scaled')
    for row in T.cells:
        for cell in row:
            x_pos, y_pos = cell.pos

            x = x_pos + (y_pos - (T.size - 1)) * 0.5
            y = y_pos
            ax.add_patch(matplotlib.patches.Circle(xy=(x, y), radius=0.45, color='brown', fill=cell.filled))

    # [p.remove() for p in reversed(ax.patches)]
    # ax.add_patch(matplotlib.patches.Circle(xy=(0, 0), radius=0.5, color='brown', fill=bool(random.getrandbits(1))))
    # ax.add_patch(matplotlib.patches.Circle(xy=(-0.5, -1), radius=0.5, color='brown', fill=bool(random.getrandbits(1))))
    # ax.add_patch(matplotlib.patches.Circle(xy=(0.5, -1), radius=0.5, color='brown', fill=bool(random.getrandbits(1))))





ani = animation.FuncAnimation(fig, animate, interval=1000)

# fig.canvas.draw()
# plt.title("Test")
# fig.canvas.draw()
#
# print(ax.patches)

plt.show()
