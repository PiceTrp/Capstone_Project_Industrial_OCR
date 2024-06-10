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
from gen_augmented_image.bbox_retriever import BoundingBoxProcessor
from gen_augmented_image.utils import *

"""
Error Take Notes
- mask mismatched >>>>>> [Done] solved by check all cases
- there's a chance that text_box is bigger than background >>>>> [Done] solved by reduce selected mask size
"""

random.seed(42)


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

    # create output dir
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_image_dir"], exist_ok=True)
    os.makedirs(config["output_bbox_dir"], exist_ok=True)

    # Augmentation start...
    for i in range(3):
        # >>> Character Creation <<<
        text = get_random_text()
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
            result_image, result_mask = insertion_processor.implement_insertion(visualize=True, verbose=True, 
                                                                                save_plot_path=os.path.join(os.getcwd(), f"insertion_{i}.jpg")) # save plot (optional)

            # save result image
            img_pil = Image.fromarray(result_image)
            img_pil.save(os.path.join(config["output_image_dir"], f"image_{i}.jpg"), quality=95)

            # save annotation
            bbox_processor = BoundingBoxProcessor(result_mask)
            bbox_processor.save_bounding_boxes_to_json(output_path=os.path.join(config["output_bbox_dir"], f"image_{i}.json"),
                                                       text_value=text)

            # save plot (optional)
            bbox_processor.plot_rectangles_on_image(save_path=os.path.join(os.getcwd(), f"bbox_{i}.jpg"))
        
        except Exception as e:
            print("An error occurred:", e)

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