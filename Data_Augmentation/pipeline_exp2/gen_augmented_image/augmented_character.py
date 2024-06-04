import numpy as np
from PIL import Image
import cv2

class AugmentedCharacter:
    def __init__(self, fake_image_path, mask_path):
        self.fake_image_path = fake_image_path
        self.mask_path = mask_path
        self.fake_image = self.load_image(fake_image_path)
        self.mask = self.get_mask(mask_path)
        self.masked = self.apply_mask()
        self.encapsulate_bbox = self.get_encapsulate_box()

    def load_image(self, path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def get_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, thresh_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological dilation to make the mask bigger to cover more detail of the fake image result
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(thresh_mask, kernel, iterations=3)
        return dilated_mask

    def apply_mask(self):
        return cv2.bitwise_and(self.fake_image, self.fake_image, mask=self.mask)

    def get_encapsulate_box(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return [x, y, w, h]

    def to_dict(self):
        return {
            "fake_image": self.fake_image,
            "bw_mask": self.mask,
            "masked": self.masked,
            "encapsulate_bbox": self.encapsulate_bbox
        }