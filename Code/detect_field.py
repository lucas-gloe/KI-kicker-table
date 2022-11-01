import cv2
import numpy as np


class DetectField:
    """
    The soccer field is determined by the position of the center spot, the
    angle of the center line and the size of the center circle.
    Since the diameter of the center circle is fixed at 20.5 cm, all other
    points of the field can be calculated by these three measures.
    """
    field = 0
    center = 0
    ratio_pxcm = 0
    angle = 0


    def get_angle(self, calibration_image):
        """
        :param calibration_image: The HSV-image to use for calculation
        :return: Rotation angle of the field in image
        """
        rgb = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
        angle = 0
        count = 0

        gray = cv2.cvtColor(cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)

        if lines.shape[0]:
            line_count = lines.shape[0]
        else:
            raise Exception('field not detected')

        for x in range(line_count):

            for rho, theta in lines[x]:
                a = np.cos(theta)
                b = np.sin(theta)
                # print(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)

                corr_angle = np.degrees(b)
                if corr_angle < 5:
                    # print(CorrAngle)
                    angle = angle + corr_angle
                    count = count + 1
                    cv2.line(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if isinstance(angle, int) and isinstance(count, int):
            angle = angle / count
            self.angle = angle
            return angle
        else:
            self.angle = 0.1
            return False

    def get_center_scale(self, calibration_image):
        """
        :param calibration_image: The HSV-image to use for calculation
        :return: Position of center point in image (tuple), ratio px per cm (reproduction scale)
        """
        gray = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=100)

        center_circle = (0, 0, 0)
        min_dist = 0xFFFFFFFFFFF
        for circle in circles[0]:
            dist_x = abs(circle[0] - calibration_image.shape[1] / 2)
            dist_y = abs(circle[1] - calibration_image.shape[0] / 2)

            if (dist_x + dist_y) < min_dist:
                min_dist = dist_x + dist_y
                center_circle = circle

        rgb = cv2.cvtColor(calibration_image, cv2.COLOR_HSV2RGB)
        cv2.circle(rgb, (int(center_circle[0]), int(center_circle[1])), int(center_circle[2]), (0, 255, 0), 1)

        center = center_circle[0], center_circle[1]
        radius = center_circle[2]
        ratio_pxcm = radius / 9.4

        self.center = center
        self.ratio_pxcm = ratio_pxcm

        return [center, ratio_pxcm]

    def calc_field(self):
        """
        This method needs some class variables. get_angle and get_center_scale
        have to be called beforehand.
        :return: field edges [Top left, top right, bottom right and bottom left corner] (list)
        """

        half_field_width = 68  # 60 + 8 for the goalkeepers feed and goal room
        half_field_height = 38  # 34 +4 for tollerance

        # x1 = int(self.center[0])
        # y1 = int(self.center[1])

        angle_radial_scale = np.radians(self.angle)

        # x2 = int((self.center[0]) + np.tan(angle_radial_scale)*(HalfFieldHeight*self.ratio_pxcm))
        # y2 = int(self.center[1] - (HalfFieldHeight*self.ratio_pxcm))

        x2 = int(self.center[0] - (half_field_width * self.ratio_pxcm) + np.tan(angle_radial_scale) *
                 (half_field_height * self.ratio_pxcm))
        y2 = int(self.center[1] - np.tan(angle_radial_scale) * (half_field_width * self.ratio_pxcm) -
                 (half_field_height * self.ratio_pxcm))
        top_left = [x2, y2]

        x2 = int(self.center[0] + (half_field_width * self.ratio_pxcm) + np.tan(angle_radial_scale) *
                 (half_field_height * self.ratio_pxcm))
        y2 = int(self.center[1] + np.tan(angle_radial_scale) * (half_field_width * self.ratio_pxcm) -
                 (half_field_height * self.ratio_pxcm))
        top_right = [x2, y2]

        x2 = int(self.center[0] - (half_field_width * self.ratio_pxcm) - np.tan(angle_radial_scale) *
                 (half_field_height * self.ratio_pxcm))
        y2 = int(self.center[1] - np.tan(angle_radial_scale) * (half_field_width * self.ratio_pxcm) +
                 (half_field_height * self.ratio_pxcm))
        bottom_left = [x2, y2]

        x2 = int(self.center[0] + (half_field_width * self.ratio_pxcm) - np.tan(angle_radial_scale) *
                 (half_field_height * self.ratio_pxcm))
        y2 = int(self.center[1] + np.tan(angle_radial_scale) * (half_field_width * self.ratio_pxcm) +
                 (half_field_height * self.ratio_pxcm))
        bottom_right = [x2, y2]

        self.field = [top_left, top_right, bottom_right, bottom_left]
        return [top_left, top_right, bottom_right, bottom_left]

    def load_game_field_properties(self, field):
        """

        """
        match_field = np.array([[int(field[0][0]), int(field[0][1])],
                                [int(field[2][0]), int(field[2][1])]])
        goal1 = np.array([[int(match_field[0][0]), int((np.linalg.norm(
            (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][1]) - 120)],
                          [int(match_field[0][0] + 70), int((np.linalg.norm(
                              (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][
                                                                 1]) + 120)]])
        goal2 = np.array([[int(match_field[1][0] - 70), int((np.linalg.norm(
            (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][1]) - 120)],
                          [int(match_field[1][0]), int((np.linalg.norm(
                              (match_field[1][1] - match_field[0][1]) / 2) + match_field[0][
                                                            1]) + 120)]])
        throw_in_zone = np.array([[int(match_field[0][0] + 400), int(match_field[0][1])],
                                  [int(match_field[1][0] - 400), int(match_field[1][1])]])

        distance_between_rods = (np.linalg.norm(match_field[1][0] - match_field[0][0])) / 8

        players_rods = np.array([[[int(match_field[0][0] + (0.5*distance_between_rods-15)+19), int(match_field[0][1])],
                                  [int(match_field[0][0] + (0.5*distance_between_rods+15)+19), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (1.5*distance_between_rods-15)+13), int(match_field[0][1])],
                                  [int(match_field[0][0] + (1.5*distance_between_rods+15)+13), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (2.5*distance_between_rods-15)+13), int(match_field[0][1])],
                                  [int(match_field[0][0] + (2.5*distance_between_rods+15)+13), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (3.5*distance_between_rods-15)+4), int(match_field[0][1])],
                                  [int(match_field[0][0] + (3.5*distance_between_rods+15)+4), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (4.5*distance_between_rods-15)-4), int(match_field[0][1])],
                                  [int(match_field[0][0] + (4.5*distance_between_rods+15)-4), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (5.5*distance_between_rods-15)-13), int(match_field[0][1])],
                                  [int(match_field[0][0] + (5.5*distance_between_rods+15)-13), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (6.5*distance_between_rods-15)-13), int(match_field[0][1])],
                                  [int(match_field[0][0] + (6.5*distance_between_rods+15)-13), int(match_field[1][1])]],
                                 [[int(match_field[0][0] + (7.5*distance_between_rods-15)-19), int(match_field[0][1])],
                                  [int(match_field[0][0] + (7.5*distance_between_rods+15)-19), int(match_field[1][1])]]])

        return [match_field, goal1, goal2, throw_in_zone, players_rods]

    def get_var(self, _type):
        """
        Get the class variables
        :param _type: String to choose the variable
        :return: The requested variable, empty string if requested name is
        unavailable
        """
        if 'field' == _type:
            return self.field
        elif 'ratio_pxcm' == _type:
            return self.ratio_pxcm
        elif 'angle' == _type:
            return self.angle
        elif 'center' == _type:
            return self.center
        else:
            return ""  # False
