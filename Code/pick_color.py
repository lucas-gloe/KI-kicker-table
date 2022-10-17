import cv2
import numpy as np

class ColorPicker:

    def check_boundaries(value, tolerance, ranges, upper_or_lower):
        if ranges == 0:
            # set the boundary for hue
            boundary = 255
        elif ranges == 1:
            # set the boundary for saturation and value
            boundary = 255

        if (value + tolerance > boundary):
            value = boundary
        elif (value - tolerance < 0):
            value = 0
        else:
            if upper_or_lower == 1:
                value = value + tolerance
            else:
                value = value - tolerance
        return value

    def pick_first_color(event, x, y, frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = frame[x, y]

            # HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
            # Set range = 0 for hue and range = 1 for saturation and brightness
            # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
            hue_upper = ColorPicker.check_boundaries(pixel[0], 10, 0, 1)
            hue_lower = ColorPicker.check_boundaries(pixel[0], 10, 0, 0)
            saturation_upper = ColorPicker.check_boundaries(pixel[1], 10, 1, 1)
            saturation_lower = ColorPicker.check_boundaries(pixel[1], 10, 1, 0)
            value_upper = ColorPicker.check_boundaries(pixel[2], 40, 1, 1)
            value_lower = ColorPicker.check_boundaries(pixel[2], 40, 1, 0)

            upper = np.array([hue_upper, saturation_upper, value_upper])
            lower = np.array([hue_lower, saturation_lower, value_lower])

            values = [upper, lower]

        return pixel, values

    def color_picker(firstframe, values, frame, pixel):
        if firstframe:
            # beim ersten frame nimm die farbe ich der user anklickt
            cv2.setMouseCallback("Video", ColorPicker.pick_first_color)
            firstframe = False
        else:
            # nimm den Punkt vom letzten frame und prüfe ob sich die Farbe verändert hat
            # nimm den punkt wo die farbe im letzten frame getracked wurde und kalibriere die farbe an diesem punkt erneut
            oldpixel = pixel

            # HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
            # Set range = 0 for hue and range = 1 for saturation and brightness
            # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
            hue_upper = ColorPicker.check_boundaries(oldpixel[0], 10, 0, 1)
            hue_lower = ColorPicker.check_boundaries(oldpixel[0], 10, 0, 0)
            saturation_upper = ColorPicker.check_boundaries(oldpixel[1], 10, 1, 1)
            saturation_lower = ColorPicker.check_boundaries(oldpixel[1], 10, 1, 0)
            value_upper = ColorPicker.check_boundaries(oldpixel[2], 40, 1, 1)
            value_lower = ColorPicker.check_boundaries(oldpixel[2], 40, 1, 0)

            upper = np.array([hue_upper, saturation_upper, value_upper])
            lower = np.array([hue_lower, saturation_lower, value_lower])

            values = [upper, lower]

        return firstframe, values, pixel



