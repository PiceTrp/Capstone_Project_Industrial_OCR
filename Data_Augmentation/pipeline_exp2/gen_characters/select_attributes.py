import os
import random

random.seed(42)


def get_random_font(fonts_dir):
    fonts = [font for font in os.listdir(fonts_dir) if font.endswith('.ttf')]
    return random.choice(fonts)


def get_proper_font_size(font_path):
    """
    Applicable for mask size of 256 only, we create bigger mask my cv2.INTER_NEAREST
    """
    if "DarkerGrotesque" in font_path:
        return 256
    elif "Raleway" in font_path:
        return 244
    elif "RedHatMono" in font_path or "SometypeMono" in font_path:
        return 244
    elif "Unison" in font_path:
        return 200
    else:
        return 200


def get_random_mask_size(candidate_mask_sizes):
    # candidate_mask_size = list of candiate mask sizes
    # the expected output result mask size
    return random.choice(candidate_mask_sizes)


def get_random_mask_padding(right_range, bottom_range):
    right_padding = random.randint(right_range[0], right_range[1])
    bottom_padding = random.randint(bottom_range[0], bottom_range[1])
    return right_padding, bottom_padding


def get_random_model_checkpoints(exp_names):
    return random.choice(exp_names)