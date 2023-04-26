import cv2
import keyboard
import os.path

import configs


def _render_field_calibrations(frame, game_config):
	"""
	show foosball field contour for calibration on frame
	Parameters:
		frame(np.ndarray):frame from interpretations
		game_config(dict): calibration values for current game
	Returns:
		frame(np.ndarray):frame with renderings
	"""
	line_color = (0, 255, 0)
	line_thickness = 2

	# Draw field contour
	field_corners = []
	for pt in game_config['field']:
		field_corners.append((int(pt[0]), int(pt[1])))
	cv2.line(frame, game_config['field'][0], game_config['field'][1], line_color, line_thickness)
	cv2.line(frame, game_config['field'][2], game_config['field'][3], line_color, line_thickness)
	cv2.line(frame, game_config['field'][0], game_config['field'][3], line_color, line_thickness)
	cv2.line(frame, game_config['field'][1], game_config['field'][2], line_color, line_thickness)

	# Draw goal1
	cv2.rectangle(frame, game_config['goal1'][0], game_config['goal1'][1], line_color, line_thickness)

	# Draw goal2
	cv2.rectangle(frame, game_config['goal2'][0], game_config['goal2'][1], line_color, line_thickness)

	# Draw throw-in zone
	cv2.rectangle(frame, game_config['throw_in_zone'][0], game_config['throw_in_zone'][1], line_color, line_thickness)

	# Draw players rods
	for rod in game_config['players_rods']:
		cv2.rectangle(frame, rod[0], rod[1], (0, 255, 255), line_thickness)

	return frame


def _render_ball(frame, results):
	"""
	draw a circle at the balls position and name the Object "ball"
	Parameters:
		frame(np.ndarray):frame from interpretations
		results(dict): dict with the current game results
	Returns:
		frame(np.ndarray):frame with renderings
	"""
	# draw a circle for the balls position
	if results['ball_position'] != [-1, -1]:
		cv2.circle(frame, (results['ball_position'][0], results['ball_position'][1]), int(16 * configs.SCALE_FACTOR),
							(0, 255, 0), 2)
		# render name from ball
		cv2.putText(frame, "Ball", (results['ball_position'][0], results['ball_position'][1]),
					cv2.FONT_HERSHEY_PLAIN, 1, (30, 144, 255), 2)
	return frame


def _render_predicted_ball(frame, results):
	"""
	draw a circle at the predicted balls position if there is no ball
	detected in the frame and name the Object "ball"
	Parameters:
		frame(np.ndarray):frame from interpretations
		results(dict): dict with the current game results
	Returns:
		frame(np.ndarray):frame with renderings
	"""
	# draw a circle on the predicted position without the name
	if results["ball_position"] == [-1, -1]:
		cv2.circle(frame, (results["predicted"][0], results["predicted"][1]), int(16 * configs.SCALE_FACTOR),
							(0, 255, 255), 2)
	return frame


def _render_figures(frame, results, team_dict_number, team_dict_name, team_number, team_ranks):
	"""
	Draw a rectangle at the players position and name it TeamX
	Parameters:
		frame(np.ndarray):frame from interpretations
		results(dict): dict with the current game results
		team_dict_number(string): team keyword in dict
		team_dict_name(string): key for boolean if team was detected
		team_number(int): team number in field
		team_ranks(string): key for list with ranks for every players postions based on their position
	Returns:
		frame(np.ndarray):frame with renderings
	"""
	if results[team_dict_name]:
		for i, player_position in enumerate(results[team_dict_number]):
			# render position
			cv2.rectangle(frame, (int(player_position[0][0]), int(player_position[0][1])),
								(int(player_position[1][0]), int(player_position[1][1])),
								(0, 255, 0), 2)
			# render name
			cv2.putText(frame,
						("Team" + str(team_number) + ", " + str(results[team_ranks][i])),
						(int(player_position[0][0]), int(player_position[0][1])), cv2.FONT_HERSHEY_PLAIN, 1,
						(30, 144, 255), 2)
	return frame


def _render_calibration_marker(frame):
	"""
	Drawing 3 circles as visible calibration markes in the middle of the frame for teams and ball calibration
	Parameters:
		frame(np.ndarray):frame from interpretations
	Returns:
		frame(np.ndarray):frame with renderings
	"""
	cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), int(18 * configs.SCALE_FACTOR),
						(30, 144, 255), 1)
	cv2.circle(frame, (int(frame.shape[1] / 2 - int(85 * configs.SCALE_FACTOR)), int(frame.shape[0] / 2)),
						int(18 * configs.SCALE_FACTOR), (30, 144, 255), 1)
	cv2.circle(frame, (int(frame.shape[1] / 2 + int(85 * configs.SCALE_FACTOR)), int(frame.shape[0] / 2)),
						int(18 * configs.SCALE_FACTOR), (30, 144, 255), 1)
	return frame


def render_game(frame, results, game_config, game_flags):
	"""
	draw all tracked objects on frame
	Parameters:
		frame(np.ndarray):frame from interpretations
		results(dict): frame results after interpretation
		game_config(dict): calibration values for current game
		game_flags(dict): flag values for current game
	Return:
		frame(np.ndarray):frame with renderings
	"""
	if results is None:
		return _render_calibration_marker(frame)
	else:
		if game_flags['show_kicker']:
			frame = _render_field_calibrations(frame, game_config)
		if game_flags['show_objects']:
			frame = _render_ball(frame, results)
			frame = _render_predicted_ball(frame, results)
			frame = _render_figures(frame, results, 'team1_positions', 'team1_on_field', 1,
													'ranks_team1')
			frame = _render_figures(frame, results, 'team2_positions', 'team2_on_field', 2,
													'ranks_team2')
		return frame


def check_variables(user_input, game_flags):
	"""
	check key bindings for user input
	Parameters:
			user_input(string): break criteria for loop
			game_flags(dict): flag values for current game
	Returns:
	"""
	variables = {
		'c': ('show_kicker', True),
		'f': ('show_kicker', False),
		'a': ('show_objects', True),
		'd': ('show_objects', False),
		'n': ('new_game', True),
		'm': ('manual_mode', True),
		'l': ('manual_mode', False),
		'k': ('one_iteration', True)
	}
	# loop over keybindings
	for key, (flag, value) in variables.items():
		if keyboard.is_pressed(key):
			user_input.value = ord(key)
			if user_input.value == ord(key):
				game_flags[flag] = value


def check_gui_events(event, current_game_results):
	"""
	check if a button on the gui was pressed
	Parameters:
		event(str): button press on gui
		current_game_results(dict): time related interpretation results for each game
	Returns:
	"""
	# check goal counter buttons
	if event in ["-manual_game_counter_team_1_up-", "-manual_game_counter_team_2_up-"]:
		if event == "-manual_game_counter_team_1_up-":
			team = "counter_team1"
		else:
			team = "counter_team2"
		current_game_results[team] += 1

	if event in ["-manual_game_counter_team_1_down-", "-manual_game_counter_team_2_down-"]:
		if event == "-manual_game_counter_team_1_down-":
			team = "counter_team1"
		else:
			team = "counter_team2"
		if current_game_results[team] > 0:
			current_game_results[team] -= 1


def update_gui(window, game_config, total_game_results, current_game_results, current_result, render_results, expect_id):
	"""
	update values on gui window
	Parameters:
		window(obj): gui window object
		game_config(dict): calibration values for current game
		total_game_results(list): time related total game results per game
		current_game_results(dict): time related interpretation results for each game
		current_result(dict): frame results after interpretation
		render_results(dict): dictionary with rendering results
		expect_id(int): frame id
	Returns:
		window(obj): gui window object
	"""
	if expect_id % 4 == 0:
		window["-frame-"].update(data=cv2.imencode('.ppm', render_results["frame"])[1].tobytes())

	window["-team_1-"].update(background_color=game_config['gui_team1_color'])
	window["-team_2-"].update(background_color=game_config['gui_team2_color'])
	window["-ball-"].update(background_color=game_config['gui_ball_color'])
	window["-score_team_1-"].update(current_game_results['counter_team1'])
	window["-score_team_2-"].update(current_game_results['counter_team2'])
	window["-fps-"].update(int(current_result['fps']))
	window["-last_game_team1-"].update(total_game_results[-1][0])
	window["-last_game_team2-"].update(total_game_results[-1][1])

	if len(total_game_results) > 1:
		window["-second_last_game_team1-"].update(total_game_results[-2][0])
		window["-second_last_game_team2-"].update(total_game_results[-2][1])
	if len(total_game_results) > 2:
		window["-third_last_game_team1-"].update(total_game_results[-3][0])
		window["-third_last_game_team2-"].update(total_game_results[-3][1])

	if os.path.exists("calibration_image.JPG"):
		window["-config_img-"].update("Konfigurationsbild gespeichert!")

	return window