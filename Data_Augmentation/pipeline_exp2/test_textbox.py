import sys
import os
import random
import yaml
from glob import glob
import shutil
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from gen_characters.gen_character import get_character_masks
from gen_augmented_image.augmented_character import AugmentedCharacterProcessor
from gen_augmented_image.create_text_image import TextBoxProcessor


# Add the path to the directory containing test.py
sys.path.append(os.path.abspath('../../Background_Removal/pytorch-CycleGAN-and-pix2pix'))


def get_random_text():
    first_position = "0123456789"
    second_position = "AHKZP"
    rest_positions = "0123456789"
    
    # Generate a random text according to the specified pattern
    random_text = random.choice(first_position) + random.choice(second_position) + \
                  ''.join(random.choice(rest_positions) for _ in range(3))
    return random_text


def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    for i in range(10):
        # create mask
        get_character_masks(get_random_text(), config)

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
        processor = TextBoxProcessor(augmented_characters, char_padding=120)
        text_box_masked, text_box_bw_mask = processor.create_text_box()

        # tempolary
        # print(os.getcwd())
        # print(config['generated_chars_dir'])
        # print(os.path.join(os.getcwd(), config['generated_chars_dir']))
        shutil.rmtree(config['generated_chars_dir'])

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(text_box_masked)
    # plt.title("Text Box Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(text_box_bw_mask, cmap="gray")
    # plt.title("Binary Mask")
    # plt.axis("off")

    # plt.show()
    


if __name__ == "__main__":
    main()