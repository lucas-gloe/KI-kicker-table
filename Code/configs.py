# define image source
# source = 1
source = r"..\Video\1080p\240fps\test_schuss_lang.mov"

# define scale factor of the videoframes and dependings
SCALE_FACTOR = 0.4

# game configs if the video source is from a video
if source != 0 or source != 1:
	HALF_WIDTH_GOAL = 130
	HALF_FIELD_WIDTH = 60 + int(30 * SCALE_FACTOR)  # 60 + 30 for the goalkeepers feed and goal room
	HALF_FIELD_HEIGHT = 34 + int(20 * SCALE_FACTOR)  # 34 + 20 for tollerance
	STREAM = False

# game configs if the video source is live
if source == 1 or source == 0:
	HALF_WIDTH_GOAL = 110
	HALF_FIELD_WIDTH = 60 + int(13 * SCALE_FACTOR)  # 60 + 13 for the goalkeepers feed and goal room
	HALF_FIELD_HEIGHT = 34 + int(10 * SCALE_FACTOR)  # 34 + 10 for tollerance
	STREAM = True

# game configs for all sources
RODWIDTH = 70 * SCALE_FACTOR
HALF_PLAYERS_WIDTH = 20 * SCALE_FACTOR
FONT = "Helvetica"

