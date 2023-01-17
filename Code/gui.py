import PySimpleGUI as sg
import os.path
import cv2

from threading import Thread


class GUI:
    """
    Class which creates and updates the gui windows for the game stats
    """

    def __init__(self, game, frame=None):
        """
        Define the gui and its depends
        """

        # creating variables and constants
        self._FONT = "Helvetica"

        self.window = None
        self.stopped = False
        self.game = game
        #self.analysis = analysis
        self.frame = frame

        # layout of the GUI

        sg.theme('Reddit')

        ################# Game frame #######################################

        game_frame = [

            [sg.Image(filename="", key="-frame-")]
        ]
       ################# left frame with basic game infos #################

        # inner frame 1

        game_score_and_speed = [
            [sg.Text('SIT Smart Kicker', text_color='orange', font=(self._FONT, 30))],
            [sg.Button('goal+1', key="-manual_game_counter_team_1_up-", button_color='grey', font=(self._FONT, 8)),sg.Button('goal-1', key="-manual_game_counter_team_1_down-", button_color='grey', font=(self._FONT, 8)), sg.Button('goal+1', key="-manual_game_counter_team_2_up-", button_color='grey', font=(self._FONT, 8)), sg.Button('goal-1', key="-manual_game_counter_team_2_down-", button_color='grey', font=(self._FONT, 8))],
            [sg.Text("", key='-score_team_1-',font=(self._FONT, 45)), sg.Text(" : ", font=(self._FONT, 20)), sg.Text("", key='-score_team_2-', font=(self._FONT, 45))],
            [sg.Text("Team 1", font=(self._FONT, 20)), sg.Text("Team 2", font=(self._FONT, 20))],
            [sg.Text("")],
            [sg.Text('Ball Speed:', font=(self._FONT, 10)), sg.Text("NOT SET YET", key='-ball_speed-', font=(self._FONT, 10))],
            [sg.Text('FPS:', font=(self._FONT, 10), text_color='grey'), sg.Text("", key='-counts_per_second-', font=(self._FONT, 10), text_color='grey')],
            [sg.Text('Press S to save configuration image',key='-config_img-', font=(self._FONT, 10))]
        ]

        # inner frame 2

        game_configuration = [
            [sg.Text("Game Config", text_color='orange', font=(self._FONT, 15))],
            [sg.Text("Team 1", font=(self._FONT, 10), text_color='white', background_color="orange", key="-team_1-", expand_x=True, justification='c')],
            [sg.Text("Team 2", font=(self._FONT, 10), text_color='white', background_color="orange", key="-team_2-", expand_x=True, justification='c')],
            [sg.Text("Ball", font=(self._FONT, 10), text_color='white', background_color="orange", key="-ball-", expand_x=True, justification='c')]
        ]

        last_games = [
            [sg.Text("Last Games", text_color='orange', font=(self._FONT, 15))],
            [sg.Text("", key='-last_game_team1-', font=(self._FONT, 10)), sg.Text(" : ", font=(self._FONT, 10)), sg.Text("", key='-last_game_team2-', font=(self._FONT, 10))],
            [sg.Text("", key='-second_last_game_team1-', font=(self._FONT, 10)), sg.Text(" : ", font=(self._FONT, 10)), sg.Text("", key='-second_last_game_team2-', font=(self._FONT, 10))],
            [sg.Text("", key='-third_last_game_team1-', font=(self._FONT, 10)), sg.Text(" : ", font=(self._FONT, 10)), sg.Text("", key='-third_last_game_team2-', font=(self._FONT, 10))]
        ]


        configs = [
            [sg.Frame("", game_configuration, expand_x=True, expand_y=True, element_justification='c'),
             sg.Frame("", last_games, expand_x=True, expand_y=True, element_justification='c')]
        ]

        # inner frame 3

        key_bindings = [
            [sg.Text("Key Bindings", text_color='orange', font=(self._FONT, 15))],
            [sg.Text('Press N to start new game', font=(self._FONT, 10))],
            [sg.Text("Press C to show kicker, press F to hide kicker",font=(self._FONT, 10))],
            [sg.Text("Press A to show contours, press D to hide contours", font=(self._FONT, 10))]
        ]

        # left frame

        basic_information = [
            [sg.Frame("", game_score_and_speed, expand_x=True, expand_y=True, element_justification='c')],
            [sg.Frame("", configs, border_width=0, expand_x=True, expand_y=True)],
            [sg.Frame("", key_bindings, expand_x=True, expand_y=True, element_justification='c')]
        ]

        game_stats = [
            [sg.Frame("", layout=basic_information, border_width=0, expand_x=True, expand_y=True)]
        ]

        ################# right frame with advanced infos #################

        # frame pattern

        heat_map = [
            [sg.Text("Heat Map", text_color='orange', font=(self._FONT, 15))]
            #[sg.Image(filename="", key="-heat_map-")]
        ]

        blank_frame = [
            [sg.Text("Place Holder", text_color='orange', font=(self._FONT, 15))]
        ]

        blank_frame2 = [
            [sg.Text("Place Holder", text_color='orange', font=(self._FONT, 15))]
        ]

        deep_information = [
            [sg.Frame("", heat_map, expand_x=True, expand_y=True, element_justification='c'), sg.Frame("", blank_frame, expand_x=True, expand_y=True, element_justification='c')],
            [sg.Frame("", blank_frame2, expand_x=True, expand_y=True, element_justification='c')]
        ]

        # right frame

        game_analysis = [
            [sg.Frame("", layout=deep_information, border_width=0, expand_x=True, expand_y=True)]
        ]

        ################# final gui layout #################

        # game_info = [
        #     [sg.Frame("Game Information", game_stats, border_width=0, size=(350, 550))],
        #     [sg.Frame("Game Statistics", game_analysis, border_width=0, size=(350, 300))]
        # ]

        # self._layout = [
        #     [sg.Frame("", game_frame, border_width=0, expand_x=True, expand_y=True, size=(1400, 700)),sg.Frame("", game_info, border_width=0, expand_x=True, expand_y=True)]
        # ]

        self._layout = [
            [sg.Frame("Game Information", game_stats, border_width=0, size=(350, 700)),
             sg.Frame("Game Statistics", game_analysis, border_width=0, size=(350, 700))]
        ]

    def start(self):
        """
        start the gui window thread
        :return: thread properties
        """
        Thread(target=self.show_gui, args=()).start()
        return self

    def stop(self):
        """
        stop and close the gui window and thread
        """
        self.stopped = True
        self.window.close()

    def show_gui(self):
        """
        interpret gui window and show the result
        """
        self.window = sg.Window('Kicker Game', self._layout)
        while not self.stopped:
            event, values = self.window.read(timeout=1)
            self.update_gui(event)
            self.window.Refresh()

    def update_gui(self, event):
        """
        update values on gui window
        """
        #self.window["-frame-"].update(data=cv2.imencode('.ppm', self.frame)[1].tobytes())
        self.window["-team_1-"].update(background_color=self.game.check_game_var("-team_1-"))
        self.window["-team_2-"].update(background_color=self.game.check_game_var("-team_2-"))
        self.window["-ball-"].update(background_color=self.game.check_game_var("-ball-"))
        self.window["-score_team_1-"].update(self.game.check_game_var("-score_team_1-"))
        self.window["-score_team_2-"].update(self.game.check_game_var("-score_team_2-"))
        # self.window["-ball_speed-"].update(self.game.check_game_var("-ball_speed-"))
        self.window["-last_game_team1-"].update(self.game.check_game_var("-last_game_team1-"))
        self.window["-last_game_team2-"].update(self.game.check_game_var("-last_game_team2-"))
        self.window["-second_last_game_team1-"].update(self.game.check_game_var("-second_last_game_team1-"))
        self.window["-second_last_game_team2-"].update(self.game.check_game_var("-second_last_game_team2-"))
        self.window["-third_last_game_team1-"].update(self.game.check_game_var("-third_last_game_team1-"))
        self.window["-third_last_game_team2-"].update(self.game.check_game_var("-third_last_game_team2-"))

        if os.path.exists("../Bilder/calibration_image.JPG"):
            self.window["-config_img-"].update("configuration image saved!")
            self.window["-counts_per_second-"].update(round(self.game.check_game_var("-counts_per_second-"), 0))
        if event == "-manual_game_counter_team_1_up-":
            self.game.check_game_var("-manual_game_counter_team_1_up-")
            self.window["-score_team_1-"].update(self.game.check_game_var("-score_team_1-"))
        if event == "-manual_game_counter_team_1_down-":
            self.game.check_game_var("-manual_game_counter_team_1_down-")
            self.window["-score_team_1-"].update(self.game.check_game_var("-score_team_1-"))
        if event == "-manual_game_counter_team_2_up-":
            self.game.check_game_var("-manual_game_counter_team_2_up-")
            self.window["-score_team_2-"].update(self.game.check_game_var("-score_team_2-"))
        if event == "-manual_game_counter_team_2_down-":
            self.game.check_game_var("-manual_game_counter_team_2_down-")
            self.window["-score_team_2-"].update(self.game.check_game_var("-score_team_2-"))

        #self.window["-heat_map-"].update(data=cv2.imencode(".ppm", self.analysis.check_analysis_var("-heat_map-")[1].tobytes()))