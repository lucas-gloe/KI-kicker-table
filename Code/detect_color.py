import cv2
import numpy as np


class ColorTracker:
    # --- color detection in a single frame --- #

    def __init__(self, scale_factor):
        self.ball_color = (-1, -1, -1)
        self.team1_color = (-1, -1, -1)
        self.team2_color = (-1, -1, -1)
        self._SCALE_FACTOR = scale_factor

    def calibrate_ball_color(self, img_hsv):
        """
        Calibration routine.
        Measures the color of the ball and stores it in the class.
        :param img_hsv: HSV-image to use for calculation.
        The ball has to be positioned in the center
        :return: None
        """
        x_center = int(round(img_hsv.shape[1] / 2))
        y_center = int(round(img_hsv.shape[0] / 2))

        # Get the color of the pixel around the image center
        colors = img_hsv[y_center - int(5 * self._SCALE_FACTOR/100):y_center + int(6 * self._SCALE_FACTOR/100), x_center - int(5 * self._SCALE_FACTOR/100):x_center + int(6 * self._SCALE_FACTOR/100)]
        lower_border_arr = [np.min(colors[:, :, 0]), np.min(colors[:, :, 1]), np.min(colors[:, :, 2])]
        upper_border_arr = [np.max(colors[:, :, 0]), np.max(colors[:, :, 1]), np.max(colors[:, :, 2])]

        # Create a mask for the areas with a color similar to the center pixel
        lower_border_arr = np.array(lower_border_arr)
        upper_border_arr = np.array(upper_border_arr)

        lower_border = tuple(lower_border_arr.tolist())
        upper_border = tuple(upper_border_arr.tolist())

        mask = cv2.inRange(img_hsv, lower_border, upper_border)

        # Average the color values of the masked area
        colors = img_hsv[mask == 255]
        h_mean = int(round(np.mean(colors[:, 0])))
        s_mean = int(round(np.mean(colors[:, 1])))
        v_mean = int(round(np.mean(colors[:, 2])))

        av = [h_mean, s_mean, v_mean]
        self.ball_color = tuple(av)

        return self.ball_color

    def recalibrate_ball_color(self, img_hsv, x_center, y_center, team1_figures, team2_figures, players_rods):
        """
        """
        team1_figures = np.array(team1_figures)
        team2_figures = np.array(team2_figures)

        checked_ball_pixels = []

        collabs = []

        # Get the color of the pixel around the image center
        ball_pixels = np.array(
            [[x_center - 5, y_center - 5], [x_center - 4, y_center - 4], [x_center - 3, y_center - 3],
             [x_center - 2, y_center - 2], [x_center - 1, y_center - 1], [x_center, y_center],
             [x_center + 5, y_center + 5], [x_center + 4, y_center + 4], [x_center + 3, y_center + 3],
             [x_center + 2, y_center + 2], [x_center + 1, y_center + 1]])

        for pixel in ball_pixels:
            for team1_figure in team1_figures:
                if (team1_figure[0][0] < pixel[0] < team1_figure[0][0]) and (
                        team1_figure[0][1] < pixel[1] < team1_figure[1][1]):
                    collabs.append(True)
                else:
                    collabs.append(False)

            for team2_figure in team2_figures:
                if (team2_figure[0][0] < pixel[0] < team2_figure[1][0]) and (
                        team2_figure[0][1] < pixel[1] < team2_figure[1][1]):
                    collabs.append(True)
                else:
                    collabs.append(False)

            for rod in players_rods:
                if (rod[0, 0] < pixel[0] < rod[1, 0]) and (rod[0, 1] < pixel[1] < rod[1, 1]):
                    collabs.append(True)
                else:
                    collabs.append(False)

            if True not in collabs:
                checked_ball_pixels.append(pixel)

            collabs = []

        checked_pixels = np.array(checked_ball_pixels)

        if len(checked_pixels) >= 1:
            checked_pixels_x = checked_pixels[:, 0]
            checked_pixels_y = checked_pixels[:, 1]

            colors = img_hsv[checked_pixels_y, checked_pixels_x]

            # colors = img_hsv[y_center - 5:y_center + 6, x_center - 5:x_center + 6] # [30:40, 60:70]
            lower_border_arr = [np.min(colors[:, 0]), np.min(colors[:, 1]), np.min(colors[:, 2])]
            upper_border_arr = [np.max(colors[:, 0]), np.max(colors[:, 1]), np.max(colors[:, 2])]

            # Create a mask for the areas with a color similar to the center pixel
            lower_border_arr = np.array(lower_border_arr)
            upper_border_arr = np.array(upper_border_arr)

            lower_border = tuple(lower_border_arr.tolist())
            upper_border = tuple(upper_border_arr.tolist())

            mask = cv2.inRange(img_hsv, lower_border, upper_border)

            # Average the color values of the masked area
            colors = img_hsv[mask == 255]
            h_mean = int(round(np.mean(colors[:, 0])))
            s_mean = int(round(np.mean(colors[:, 1])))
            v_mean = int(round(np.mean(colors[:, 2])))

            av = [h_mean, s_mean, v_mean]
            self.ball_color = tuple(av)

            return self.ball_color

        else:
            no_adaption = [0, 0, 0]
            return no_adaption

    def calibrate_team_color(self, img_hsv, team_number):
        """
        Calibration routine.
        Measures the color of the ball and stores it in the class.
        :param img_hsv: HSV-image to use for calculation.
        The ball has to be positioned in the center
        :return: None
        """
        # Get the exact point for measuring
        x_center = int(round(img_hsv.shape[1] / 2))
        y_center = int(round(img_hsv.shape[0] / 2))

        if team_number == 1:
            x_player = x_center + int(85 * self._SCALE_FACTOR / 100)
            y_player = y_center

        if team_number == 2:
            x_player = x_center - int(85 * self._SCALE_FACTOR / 100)
            y_player = y_center

        # Get the color of the pixel in the image center
        color = img_hsv[y_player, x_player]
        colors = img_hsv[y_player - int(6 * self._SCALE_FACTOR/100):y_player + int(5* self._SCALE_FACTOR/100), x_player - int(6* self._SCALE_FACTOR/100):x_player + int(5* self._SCALE_FACTOR/100)]
        lower_border_arr = [np.min(colors[:, :, 0]), np.min(colors[:, :, 1]), np.min(colors[:, :, 2])]
        upper_border_arr = [np.max(colors[:, :, 0]), np.max(colors[:, :, 1]), np.max(colors[:, :, 2])]

        # Create a mask for the areas with a color similar to the center pixel
        lower_border_arr = np.array(lower_border_arr)
        upper_border_arr = np.array(upper_border_arr)

        lower_border = tuple(lower_border_arr.tolist())
        upper_border = tuple(upper_border_arr.tolist())

        mask = cv2.inRange(img_hsv, lower_border, upper_border)

        # Average the color values of the masked area
        colors = img_hsv[mask == 255]
        h_mean = int(round(np.mean(colors[:, 0])))
        s_mean = int(round(np.mean(colors[:, 1])))
        v_mean = int(round(np.mean(colors[:, 2])))

        av = [h_mean, s_mean, v_mean]
        if team_number == 1:
            self.team1_color = tuple(av)
            return self.team1_color

        if team_number == 2:
            self.team2_color = tuple(av)
            return self.team2_color
