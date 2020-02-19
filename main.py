import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from actor import Actor
from actor_critic_agent import ActorCriticAgent
from critic import Critic
from neural_critic import NeuralCritic
from pivotals import *

matplotlib.use("TkAgg")


class Visualizer:

    def __init__(self):
        self.board = BOARD(BOARD_SIZE, EMPTY_CELLS)
        self.actor = Actor(INITIAL_EPSILON,
                           EPSILON_DECAY_RATE,
                           LEARN_RATE_ACTOR,
                           DISCOUNT_FACTOR_ACTOR,
                           ELIGIBILITY_DECAY_ACTOR
                           )

        self.critic = self._init_critic(USE_NN)
        self.agent = ActorCriticAgent(self.actor, self.critic, self.board)
        self.num_pegs = []

    def _init_critic(self, use_neural_network):
        if use_neural_network:
            return NeuralCritic(LEARN_RATE_CRITIC,
                                ELIGIBILITY_DECAY_CRITIC,
                                DISCOUNT_FACTOR_CRITIC,
                                input_shape=len(self.board.state),
                                hidden_shapes=HIDDEN_LAYER_NODES
                                )
        else:
            return Critic(LEARN_RATE_CRITIC,
                          ELIGIBILITY_DECAY_CRITIC,
                          DISCOUNT_FACTOR_CRITIC
                          )

    def animation(self, num_games):
        self.num_pegs, _ = self.agent.play_many(num_games)
        self.actor.epsilon = 0
        _, action_sequence = self.agent.play_many(1)

        fig, (board_ax, graph_ax) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)

        graph_ax.plot(self.num_pegs)
        board_ax.axes.get_xaxis().set_visible(True)
        board_ax.axes.get_yaxis().set_visible(True)
        y_max = 0.5
        y_min = -((self.board.size - 1) * 2 ** 0.5 + 0.5)
        x_max = ((self.board.size - 1) * 2 ** 0.5) / 2 + 0.5
        x_min = -((self.board.size - 1) * 2 ** 0.5) / 2 - 0.5

        def animate(i):
            if i > 2 * len(action_sequence[-1]) - 1:
                return
            if i % 2:
                self.board.apply_action(action_sequence[-1][i // 2])
            else:
                self.board.visualize_action(action_sequence[-1][i // 2])

            board_ax.clear()
            board_ax.set_ylim(y_min, y_max)
            board_ax.set_xlim(x_min, x_max)
            edges = self.board.get_neighbour_pairs()
            for edge in edges:
                x_start = edge[0].pos[0]
                y_start = edge[0].pos[1]
                x_end = edge[1].pos[0]
                y_end = edge[1].pos[1]
                board_ax.plot([x_start, x_end], [y_start, y_end], "k-", zorder=1)  # Draw black lines between cells
            for row in self.board.cells:
                for cell in row:
                    board_ax.add_patch(
                        matplotlib.patches.Circle(
                            xy=cell.pos,
                            radius=0.45,
                            facecolor=cell.color,
                            edgecolor="brown",
                            fill=True,
                            zorder=2
                        )
                    )
                    board_ax.text(x=cell.pos[0], y=cell.pos[1], s=cell.index, fontsize=12)

        ani = animation.FuncAnimation(fig, animate, interval=FRAME_DELAY, blit=False)
        plt.show()

    def plot_detailed_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10, 5)
        averages = []
        acc = 0
        for i, elem in enumerate(self.num_pegs, 1):
            acc += elem
            averages.append(acc / i)
        ax1.plot(self.num_pegs, label="num pegs")
        ax1.plot(averages, label="average")
        ax1.plot(self.actor.epsilon_sequence, label="epsilon")
        ax1.plot([0] * len(self.critic.td_errors), "black")
        ax2.plot(self.critic.td_errors, label="TD errors")
        avg = []
        acc = 0
        for i, elem in enumerate(self.critic.td_errors, 1):
            acc += elem
            avg.append(acc / i)
        ax2.plot([0] * len(self.critic.td_errors), "black")
        ax1.legend()
        ax2.legend()
        plt.show()


if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.animation(EPISODES)
    if DETAILED_PLOT:
        visualizer.plot_detailed_results()
