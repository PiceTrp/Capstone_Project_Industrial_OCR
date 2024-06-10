import os
import sys

import yaml
from glob import glob
import shutil
from tqdm import tqdm

from gen_characters.gen_character import get_character_masks
from gen_augmented_image.augmented_character import AugmentedCharacterProcessor
from gen_augmented_image.create_text_image import TextBoxProcessor
from gen_augmented_image.non_character_background import NonCharacterBackgroundProcessor
from gen_augmented_image.insertion import Insertion
from gen_augmented_image.bbox_retriever import BoundingBoxProcessor
from gen_augmented_image.utils import *

"""
Error Take Notes
- mask mismatched >>>>>> [Done] solved by check all cases
- there's a chance that text_box is bigger than background >>>>> [Done] solved by reduce selected mask size
"""

# Add the path to the directory containing test.py
sys.path.append(os.path.abspath('../../Background_Removal/pytorch-CycleGAN-and-pix2pix'))

# set seed
random.seed(42)


def get_random_text():
    first_position = "34689" # for super specific to our case
    second_position = "AHKZP"
    third_position = "0"
    rest_positions = "0123456789"
    
    # Generate a random text according to the specified pattern
    random_text = random.choice(first_position) + random.choice(second_position) + third_position + \
                  ''.join(random.choice(rest_positions) for _ in range(2))
    return random_text


def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    # create output dir
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_image_dir"], exist_ok=True)
    os.makedirs(config["output_mask_dir"], exist_ok=True)
    os.makedirs(config["output_bbox_dir"], exist_ok=True)

    # Augmentation start...
    for i in tqdm(range(4000)):
        # >>> Character Creation <<<
        text = get_random_text()
        print(f"create text... {text}")
        get_character_masks(text, config)

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
        background_processor = NonCharacterBackgroundProcessor(background_images[random.randint(0,len(background_images)-1)])

        # Create text box
        text_box_processor = TextBoxProcessor(augmented_characters)

        try:
            # test insertion
            insertion_processor = Insertion(background_processor, text_box_processor)
            result_image, result_mask = insertion_processor.implement_insertion(verbose=True)

            # save result image
            img_pil = Image.fromarray(result_image)
            img_pil.save(os.path.join(config["output_image_dir"], f"image_{i}.jpg"), quality=95)
            print("save image...")

            mask_pil = Image.fromarray(result_mask)
            mask_pil.save(os.path.join(config["output_mask_dir"], f"mask_{i}.jpg"), quality=95)
            print("save mask...")

            # save annotation
            bbox_processor = BoundingBoxProcessor(result_mask)
            bbox_processor.save_bounding_boxes_to_json(output_path=os.path.join(config["output_bbox_dir"], f"image_{i}.json"),
                                                    text_value=text)
            print("save annotation...")
        except Exception as e:
            print("An error occurred:", e)

        # for the end of everything
        shutil.rmtree(config['generated_chars_dir'])


if __name__ == "__main__":
    main()