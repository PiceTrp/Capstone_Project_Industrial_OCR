import numpy as np
from PIL import Image
import cv2
import random


def get_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, thresh_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply morphological dilation to make the mask bigger to cover more detail of the fake image result
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(thresh_mask, kernel, iterations=3)
    return dilated_mask


def get_encapsulate_box(dilated_mask):
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h


def get_augmented_character(fake_image_path, mask_path):
    fake_image = np.array(Image.open(fake_image_path).convert("RGB"), dtype=np.uint8)
    mask = get_mask(mask_path) # dilated_mask
    masked = cv2.bitwise_and(fake_image, fake_image, mask=mask)
    x, y, w, h = get_encapsulate_box(mask)
    # result in dictionary
    augmented_character = {"fake_image":fake_image, "bw_mask":mask, "masked": masked, "encapsulate_bbox":[x, y, w, h]}
    return augmented_character


def get_text_box_size(augmented_characters, total_char=5, char_padding=30):
    """
    Calculate The Total area used to place 5 selected characters
    """
    widths, heights = [], []
    for augmented_character in augmented_characters:
        x,y,w,h = augmented_character["encapsulate_bbox"]
        widths.append(w)
        heights.append(h)
    width = sum(widths) + char_padding*(total_char-1)
    height = max(heights)
    return (width, height)


def get_cropped_character(augmented_characters, char_index):
    """
    get the cropped version of bw_mask & masked on a single character
    """
    x,y,w,h = augmented_characters[char_index]["encapsulate_bbox"]
    bw_mask = augmented_characters[char_index]["bw_mask"]
    cropped_bw_mask = bw_mask[y:y+h, x:x+w]
    masked = augmented_characters[char_index]["masked"]
    cropped_masked = masked[y:y+h, x:x+w]
    return cropped_bw_mask, cropped_masked


def get_paired_cropped_characters(augmented_characters):
    """
    get list of cropped version of bw_mask & masked on all characters in augmented_characters
    """
    cropped_bw_masks = []
    cropped_masked_s = []
    for char_index in range(len(augmented_characters)):
        cropped_bw_mask, cropped_masked = get_cropped_character(augmented_characters, char_index)
        cropped_bw_masks.append(cropped_bw_mask)
        cropped_masked_s.append(cropped_masked)
    return cropped_bw_masks, cropped_masked_s


def add_obj(background, img, mask, x, y):
    '''
    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in GRAY-SCALE format
    x, y - coordinates of the top-left of the object image
    0 < x < width of background
    0 < y < height of background

    Function returns background with added object in CV2 RGB format !!!! RGB !!!!
    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''
    bg = background.copy()
    # width & height of the text box
    h_bg, w_bg = bg.shape[0], bg.shape[1]

    # width & height of the character mask
    h, w = img.shape[0], img.shape[1]

    mask_boolean = mask[:,:,0] == 255 # black == True !!!!!!
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

    # REQUIRED: x >= 0 and y >= 0
    # place single character on text box
    # x,y = top-left coordinate !!!!
    h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
    w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis
    bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
    return bg


def create_text_box(augmented_characters, char_padding=30):
    """
    output the merged characters to be a single text box containing 5 characters
    output will be (text box of masked_s, text box of bw_masks)
    """
    # get the text box canva
    text_box_w, text_box_h = get_text_box_size(augmented_characters,
                                           total_char=len(augmented_characters),
                                           char_padding=char_padding) # !!
    text_box_image = np.zeros((text_box_h, text_box_w, 3), dtype=np.uint8)
    # get text box for both text_box_masked & text_box_bw_mask
    text_box_masked = text_box_image.copy()
    text_box_bw_mask = text_box_image.copy()
    # size of text box, same for both text_box_masked & text_box_bw_mask
    h_bg, w_bg = text_box_image.shape[0], text_box_image.shape[1]

    # get masks & masked_s
    cropped_bw_masks, cropped_masked_s = get_paired_cropped_characters(augmented_characters)

    """
    Each character will be placed next to each on the right + have "char_padding" distance to each
    So we need to calculate the top-left coordinates to correctly place each character
    """
    # start calculate top-left coordinate - first character top-left will be at (0,0)
    x = 0
    y = 0 # this will be fixed cause we shift character through x-axis

    for index, (bw_mask, masked) in enumerate(zip(cropped_bw_masks, cropped_masked_s)):
        """
        masked  = stylized character image
        bw_mask = black-white character mask
        """
        # get width & height of the character
        h, w = bw_mask.shape[0], bw_mask.shape[1]
        # convert bw_mask to rgb before use
        bw_mask = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2RGB)
        # place on bg_masked
        text_box_masked = add_obj(text_box_masked, masked, bw_mask, x, y)
        # place on bg_bw_mask
        text_box_bw_mask = add_obj(text_box_bw_mask, bw_mask, bw_mask, x, y)

        # get correct top-left coordinate for the next character
        if index < len(cropped_bw_masks)-1:
            x = x + w + char_padding
            
    return text_box_masked, text_box_bw_mask


def get_insert_area(bg_image):
    """
    to return coordinates of the mask insertion area for placing text box containing fake characters
    Step: thresholding + find largest contour + get coordinates
    >> Input: Non-character Background image
    >> Output: (largest contour coordinates, top-left & bottom-right of mask insertion area)
    """
    # Otsu
    if isinstance(bg_image, str): # image path
        image = cv2.imread(bg_image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(bg_image, np.ndarray):
        image = bg_image # image array
    else:
        print("Failed")
        return None

    # Otsu thresholding
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the largest contour by sorting contours based on area
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the rectangle that encapsulate the largest contour = Mask Insertion Area
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_rect_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

    # get top-left & bottom-right of mask insertion area
    placable_topleft = bounding_rect_contour[0]
    placable_bottomright = bounding_rect_contour[2]
    return largest_contour, placable_topleft, placable_bottomright


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