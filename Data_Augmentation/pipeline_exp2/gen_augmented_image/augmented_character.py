import cv2
import numpy as np
from PIL import Image

class AugmentedCharacterProcessor:
    def __init__(self, fake_image_path, mask_path):
        self.fake_image_path = fake_image_path
        self.mask_path = mask_path

    @staticmethod
    def get_mask(mask_path):
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, thresh_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological dilation to make the mask bigger to cover more detail of the fake image result
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(thresh_mask, kernel, iterations=3)
        return dilated_mask

    @staticmethod
    def get_encapsulate_box(dilated_mask):
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h

    def get_augmented_character(self):
        fake_image = np.array(Image.open(self.fake_image_path).convert("RGB"), dtype=np.uint8)
        mask = self.get_mask(self.mask_path)  # dilated_mask
        masked = cv2.bitwise_and(fake_image, fake_image, mask=mask)
        x, y, w, h = self.get_encapsulate_box(mask)
        # result in dictionary
        augmented_character = {"fake_image": fake_image, "bw_mask": mask, "masked": masked, "encapsulate_bbox": [x, y, w, h]}
        return augmented_character
