# source = 1
source = r"C:\Users\gloec\OneDrive\Dokumente\GitHub\KI-kicker-table\OpenCV\Video\1080p\240fps\test_schuss_lang.mov"

SCALE_FACTOR = 0.4

if source != 0 or source != 1:
    HALF_WIDTH_GOAL = 130
    HALF_FIELD_WIDTH = 60 + int(30 * SCALE_FACTOR)  # 60 + 8 for the goalkeepers feed and goal room
    HALF_FIELD_HEIGHT = 34 + int(20 * SCALE_FACTOR)  # 34 +4 for tollerance
    STREAM = False

if source == 1 or source == 0:
    HALF_WIDTH_GOAL = 110
    HALF_FIELD_WIDTH = 60 + int(13 * SCALE_FACTOR)  # 60 + 8 for the goalkeepers feed and goal room
    HALF_FIELD_HEIGHT = 34 + int(10 * SCALE_FACTOR)  # 34 +4 for tollerance
    STREAM = True

RODWIDTH = 70 * SCALE_FACTOR
HALF_PLAYERS_WIDTH = 20 * SCALE_FACTOR

FONT = "Helvetica"

