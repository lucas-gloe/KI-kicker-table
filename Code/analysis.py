import numpy as np
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns
from threading import Thread


class Analysis:
    """
    Class that uses tracked date from game class to analyse the game
    """

    def __int__(self, game=None):
        """
        creating variables and constants
        """
        self.game = game

    def start(self):
        Thread(target=self.calculate_game_analysis, args=()).start()

    def calculate_game_analysis(self):
        self._create_heat_map()
        # self.create_some_game_statistic

    def _create_heat_map(self):

        data = [[10, 20], [20, 20], [50, 30], [30, 25], [26, 23], [25, 45], [25, 25], [25, 31], [80, 80], [42, 52],
               [32, 32], [20, 26], [45, 45]]
        #data=self.game.ball_positions
        data = np.array(data)

        fig, ax = plt.subplots(figsize=(13, 8.5))

        pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
        pitch.draw(ax=ax)

        kde = sns.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            fill=True,
            shade=True,
            shade_lowest=False,
            n_levels=100,
            alpha=0.3,
            cmap="magma"
        )

        #return fig.show()

    def check_analysis_var(self, _type):
        if "-heat_map-" == _type:
            return self._create_heat_map()
