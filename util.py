import numpy as np


def frame_norm(frame, bbox):
    # convert normalized bbox values into actual pixel(frame) positions
    return (
        np.array(bbox * np.array([*frame.shape[:2], *frame.shape[:2]][::-1]))
    ).astype(int)
