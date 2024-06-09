import os
import random
import yaml
from glob import glob
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from gen_characters.gen_character import get_character_masks
from gen_augmented_image.augmented_character import AugmentedCharacterProcessor
from gen_augmented_image.create_text_image import TextBoxProcessor
from gen_augmented_image.non_character_background import NonCharacterBackgroundProcessor
from gen_augmented_image.utils import *

def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    # >>> Insertion Implementation <<<
    fake_image_paths = sorted(glob(os.path.join(config['generated_chars_image_dir'], "*fake*")))
    mask_paths = sorted(glob(os.path.join(config['generated_chars_image_dir'], "*real*")))
    # bg_image = np.array(Image.open(config["background_image_path"]).convert("L"))

    augmented_characters = []
    for fake_image_path, mask_path in tqdm(zip(fake_image_paths, mask_paths)):
        processor = AugmentedCharacterProcessor(fake_image_path, mask_path)
        augmented_character = processor.get_augmented_character()
        augmented_characters.append(augmented_character)

    #Create text box
    processor = TextBoxProcessor(augmented_characters, char_padding=30)
    text_box_masked, text_box_bw_mask = processor.create_text_box()

    # Create an instance of the NonCharacterBackground class
    background_images = sorted(glob(os.path.join(config['background_dir'], "*.png")))
    background_processor = NonCharacterBackgroundProcessor(background_images[2])
    background_image = background_processor.get_cropped_background()
    background_insertion_region_mask = background_processor.get_cropped_combined_mask()

    # get top-left & bottom-right of mask insertion area
    placable_topleft, placable_bottomright = get_insert_area(background_insertion_region_mask)
    print(placable_topleft, placable_bottomright)

    


    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(background_image)
    # plt.title("BG Image")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(background_mask, cmap="gray")
    # plt.title("Binary Mask")
    # plt.axis("off")

    # plt.subplot(1, 3, 3)
    # plt.imshow(result)
    # plt.title("Inserted")
    # plt.axis("off")

    # plt.show()
    


if __name__ == "__main__":
    main()