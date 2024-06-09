import os
import yaml
from glob import glob
import shutil
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from gen_characters.gen_character import get_character_masks
from gen_augmented_image.augmented_character import AugmentedCharacterProcessor
from gen_augmented_image.create_text_image import TextBoxProcessor
from gen_augmented_image.non_character_background import NonCharacterBackgroundProcessor
from gen_augmented_image.insertion import Insertion
from gen_augmented_image.utils import *

def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    # >>> Character Creation <<<
    get_character_masks("8A107", config)

    # >>> Insertion Implementation <<<
    fake_image_paths = sorted(glob(os.path.join(config['generated_chars_image_dir'], "*fake*")))
    mask_paths = sorted(glob(os.path.join(config['generated_chars_image_dir'], "*real*")))
    background_images = sorted(glob(os.path.join(config['background_dir'], "*.png")))

    augmented_characters = []
    for fake_image_path, mask_path in tqdm(zip(fake_image_paths, mask_paths)):
        processor = AugmentedCharacterProcessor(fake_image_path, mask_path)
        augmented_character = processor.get_augmented_character()
        augmented_characters.append(augmented_character)

    # Create an instance of the NonCharacterBackground class
    background_processor = NonCharacterBackgroundProcessor(background_images[2])
    # Create text box
    text_box_processor = TextBoxProcessor(augmented_characters, char_padding=30)
    overlay_value = 20

    # test insertion
    insertion_processor = Insertion(background_processor, text_box_processor, overlay_value)
    result_image = insertion_processor.implement_insertion(visualize=True)
    plt.imshow(result_image)
    plt.show()

    # for the end of everything
    shutil.rmtree(config['generated_chars_dir'])

    
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(background_image)
    # plt.title("BG Image")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(insertion_mask, cmap="gray")
    # plt.title("Binary Mask")
    # plt.axis("off")

    # bg_copy = np.zeros(background_image.shape, dtype=np.uint8)
    # cv2.line(bg_copy, (0, placable_topleft[1]), (bg_copy.shape[1], placable_topleft[1]), (12, 242, 223), 10)
    # cv2.line(bg_copy, (0, placable_bottomright[1]), (bg_copy.shape[1], placable_bottomright[1]), (12, 242, 223), 10)
    # plt.subplot(1, 3, 3)
    # plt.imshow(bg_copy)
    # plt.title("placable coordinates")
    # plt.axis("off")

    # plt.show()
    


if __name__ == "__main__":
    main()