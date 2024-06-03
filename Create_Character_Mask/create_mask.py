import os
import random
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

def get_random_font_path(config):    
    fonts_dir = config["fonts_dir"]
    fonts = [os.path.join(fonts_dir, font) for font in os.listdir(fonts_dir) if font.endswith('.ttf')]
    return random.choice(fonts)


def get_random_mask_padding(config):
    right_range = config["right_padding"]
    bottom_range = config["bottom_padding"]
    right_padding = random.randint(right_range[0], right_range[1])
    bottom_padding = random.randint(bottom_range[0], bottom_range[1])
    return right_padding, bottom_padding


def get_proper_font_size(font_path):
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


def create_character_mask(char, pic_index, config):
    image = Image.new("RGB", (config["mask_size"], config["mask_size"]), "black")
    font_path = get_random_font_path(config)
    font_size = get_proper_font_size(font_path)
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)

    # draw
    _, _, w, h = draw.textbbox((0, 0), char, font=font)
    right_padding, bottom_padding = get_random_mask_padding(config)
    c_w, c_h = (config["mask_size"]-w)/2 - right_padding, (config["mask_size"]-h)/2 - bottom_padding
    draw.text((c_w, c_h), char, fill="white", font=font)  # draw text mask at CENTER

    # save created mask
    if not os.path.exists(config["mask_created_save_dir"]):
        os.makedirs(config["mask_created_save_dir"], exist_ok=True)
    image.save(os.path.join(config["mask_created_save_dir"], f"{pic_index}_{os.path.basename(font_path).replace('.ttf', '')}_{char}.png"))  # save