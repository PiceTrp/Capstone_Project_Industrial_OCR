import os
from PIL import Image, ImageDraw, ImageFont

# relative import
from .select_attributes import get_proper_font_size, get_random_mask_padding


def create_character_mask(char, font, mask_size, pic_index, config):
    # create with 256 because every configuration for proper output based on size of 256, 
    # then scale later according to its selected mask_size
    # configuration = fixed proper font_size for each different font family

    # create mask size of 256
    image = Image.new("RGB", (256, 256), "black")
    font_path = os.path.join(config["fonts_dir"], font)
    font_size = get_proper_font_size(font_path)
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)

    # draw
    _, _, w, h = draw.textbbox((0, 0), char, font=font)
    right_padding, bottom_padding = get_random_mask_padding(config["right_padding_range"], config["bottom_padding_range"])
    c_w, c_h = (256-w)/2 - right_padding, (256-h)/2 - bottom_padding
    draw.text((c_w, c_h), char, fill="white", font=font)  # draw text mask at CENTERs

    # scale result image according to its selected mask size using NEAREST
    scaled_image = image.resize((mask_size, mask_size), Image.NEAREST)

    # save created mask
    if not os.path.exists(config["mask_created_save_dir"]):
        os.makedirs(config["mask_created_save_dir"], exist_ok=True)
    scaled_image.save(os.path.join(config["mask_created_save_dir"], f"{pic_index}_{os.path.basename(font_path).replace('.ttf', '')}_{char}.png"))  # save