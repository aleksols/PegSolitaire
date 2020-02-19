# import matplotlib
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
#
# from actor import Actor
# from actor_critic_agent import ActorCriticAgent
# from critic import Critic
# from neural_critic import NeuralCritic
# from pivotals import FRAME_DELAY, EPISODES_BEFORE_VISUALIZATION, USE_NN
#
# matplotlib.use("TkAgg")
#
# actor = Actor()
# if USE_NN:
#     critic = NeuralCritic()
# else:
#     critic = Critic()
#
# agent = ActorCriticAgent(actor, critic)
#
# rewards, results, _ = agent.play_many(EPISODES_BEFORE_VISUALIZATION)
# print(actor.epsilon)
# actor.epsilon = 0
# reward, num_pegs, actions = agent.play_many(1)
#
#
# board = agent.environment
#
# fig, (board_ax, graph_ax) = plt.subplots(1, 2)
# fig.set_size_inches(12, 6)
#
# graph_ax.plot(results)
# board_ax.axes.get_xaxis().set_visible(True)
# board_ax.axes.get_yaxis().set_visible(True)
# y_max = 0.5
# y_min = -((board.size - 1) * 2 ** 0.5 + 0.5)
# x_max = ((board.size - 1) * 2 ** 0.5) / 2 + 0.5
# x_min = -((board.size - 1) * 2 ** 0.5) / 2 - 0.5
#
#
# def animate(i):
#     if i > 2 * len(actions[-1]) - 1:
#         return
#     if i % 2:
#         board.apply_action(actions[-1][i // 2])
#     else:
#         board.visualize_action(actions[-1][i // 2])
#
#     board_ax.clear()
#     board_ax.set_ylim(y_min, y_max)
#     board_ax.set_xlim(x_min, x_max)
#     edges = board.get_neighbour_pairs()
#     for edge in edges:
#         x_start = edge[0].pos[0]
#         y_start = edge[0].pos[1]
#         x_end = edge[1].pos[0]
#         y_end = edge[1].pos[1]
#         board_ax.plot([x_start, x_end], [y_start, y_end], "k-", zorder=1)  # Draw black lines between cells
#     for row in board.cells:
#         for cell in row:
#             board_ax.add_patch(
#                 matplotlib.patches.Circle(
#                     xy=cell.pos,
#                     radius=0.45,
#                     facecolor=cell.color,
#                     edgecolor="brown",
#                     fill=True,
#                     zorder=2
#                 )
#             )
#             board_ax.text(x=cell.pos[0], y=cell.pos[1], s=cell.index, fontsize=12)
#
#
# ani = animation.FuncAnimation(fig, animate, interval=FRAME_DELAY, blit=False)
# plt.show()
