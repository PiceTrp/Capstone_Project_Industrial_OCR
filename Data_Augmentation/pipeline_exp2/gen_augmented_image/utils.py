import numpy as np
from PIL import Image
import cv2
import random


def get_random_coordinate(placable_topleft, placable_bottomright, text_box_masked):
    x1,y1 = placable_topleft
    x2,y2 = placable_bottomright
    h, w = text_box_masked.shape[:-1]
    # find range that is able to randomly place the top-left coordinate of text box on the background image
    # range for top-left only
    random_width_range = (x1, x2-w)
    random_height_range = (y1, y2-h)

    random_x = random.randint(random_width_range[0], random_width_range[1])
    random_y = random.randint(random_height_range[0], random_height_range[1])

    return (random_x, random_y)

# bg_image input of this funciton will be [ gray-scale ] !!!!
def randomly_place_text_box(bg_image, text_box_masked, text_box_bw_mask):
    # get_insert_area needs bg to be gray-scale
    _, placable_topleft, placable_bottomright = get_insert_area(bg_image)
    random_x, random_y = get_random_coordinate(placable_topleft, placable_bottomright, text_box_masked)

    # add_obj need bg to be rgb
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2RGB)
    result_image = add_obj(bg_image, text_box_masked, text_box_bw_mask, random_x, random_y)
    return result_image, (random_x, random_y)