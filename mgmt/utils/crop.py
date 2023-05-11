import numpy as np


def slide_box_within_border(
    center_x,
    center_y,
    image_width,
    image_height,
    box_width=40,
    box_height=None,
):
    width_half = int(box_width / 2)
    if box_height is None:
        height_half = width_half
    else:
        height_half = int(box_height / 2)

    # bound the center point
    min_x = width_half
    max_x = image_width - width_half
    min_y = height_half
    max_y = image_height - height_half
    center_x = int(np.round(center_x))
    center_y = int(np.round(center_y))
    center_x = max(center_x, min_x)
    center_x = min(center_x, max_x)
    center_y = max(center_y, min_y)
    center_y = min(center_y, max_y)
    return center_x, center_y
