import sys
import os
import random
import yaml
from glob import glob
import numpy as np
import cv2

from gen_characters.gen_character import get_character_masks
from gen_augmented_image.augmented_character import AugmentedCharacterProcessor

def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    # >>> Insertion Implementation <<<
    print(glob(os.path.join(config["generated_chars_dir"], "**", "test_latest", "*fake*").replace("\\", "/")))

    fake_image_paths = sorted(glob(os.path.join(config["generated_chars_dir"], "**", "test_latest", "*fake*")))
    mask_paths = sorted(glob(os.path.join(config["generated_chars_dir"], "**", "test_latest", "*real*")))
    # bg_image = np.array(Image.open(config["background_image_path"]).convert("L"))
    print(fake_image_paths, mask_paths)

    augmented_characters = []
    for fake_image_path, mask_path in zip(fake_image_paths, mask_paths):
        processor = AugmentedCharacterProcessor(fake_image_path, mask_path)
        augmented_character = processor.get_augmented_character()
        augmented_characters.append(augmented_character)

    # Display one of the results
    augmented_character = augmented_characters[0]
    cv2.imshow("Fake Image", augmented_character["fake_image"])
    cv2.imshow("Binary Mask", augmented_character["bw_mask"])
    cv2.imshow("Masked Image", augmented_character["masked"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()