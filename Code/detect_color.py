import cv2
import numpy as np


class ColorTracker:
    # --- color detection in a single image --- #

    ball_detection_threshold = 0.2
    interface = 0

    ball_color = (-1, -1, -1)
    team1_color = (-1, -1, -1)
    team2_color = (-1, -1, -1)
    curr_ball_position = (-1, -1)

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

        # Get the color of the pixel in the image center
        color = img_hsv[y_center, x_center]

        colors = img_hsv[y_center - 5:y_center + 6, x_center - 5:x_center + 6]
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
            x_player = x_center + 85
            y_player = y_center

        if team_number == 2:
            x_player = x_center - 85
            y_player = y_center


        # Get the color of the pixel in the image center
        color = img_hsv[y_player, x_player]
        colors = img_hsv[y_player - 5:y_player + 6, x_player - 5:x_player + 6]
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

    def _check_circle(self, points):
        """
        Calculates a comparison value with a circle.
        First, it normalizes the given points, so that their mean is the origin and their distance to the origin
        is 1 in average.
        Then it averages the differences between the points' distance to the origin and 1.
        The resulting value is 0 when the points form a circle, and increases, if there is any deformation.
        It has no upper limit, but will not be smaller than 0.
        To sum up: the lower the value, the better fit the points to a circle
        :param points: the points that mark the contour to check
        :return: Comparison value.
        """
        # Split x- and y-Values into two arrays
        x_vals, y_vals = [], []
        for point in points:
            x_vals.append(point[0])
            y_vals.append(point[1])

        # Shift the circle center to (0,0)
        x_vals = x_vals - np.mean(x_vals)
        y_vals = y_vals - np.mean(y_vals)

        # Bring the circle radius to 1
        radius = np.sqrt((np.sum(x_vals ** 2 + y_vals ** 2)) / len(x_vals))
        for point in range(0, len(x_vals)):
            x_vals[point] = x_vals[point] / radius
            y_vals[point] = y_vals[point] / radius

        # Now the result is compared to a unit circle (radius 1), and the
        # differences are averaged.
        dist = np.mean(np.abs(x_vals ** 2 + y_vals ** 2 - 1))

        return dist


    def get_var(self, _type):
        """
        Get the class variables
        :param _type: String to choose the variabe
        :return: The requested variable, empty string if requested name is
        unavailable
        """
        if 'ball_position' == _type:
            return self.curr_ball_position
        else:
            return ""  # False
