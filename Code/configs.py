video_stream = False
video_pic = False
video_vid = True

def get_frame_by_frame():
    if video_vid:
        return True
    if video_pic:
        return True
    else:
        return False