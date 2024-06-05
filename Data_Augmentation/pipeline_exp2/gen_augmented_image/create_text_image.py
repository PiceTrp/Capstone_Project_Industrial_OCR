import cv2
import numpy as np

class TextBoxProcessor:
    def __init__(self, augmented_characters, char_padding=30):
        self.augmented_characters = augmented_characters
        self.char_padding = char_padding

    def get_text_box_size(self):
        """
        Calculate the total area used to place the selected characters.
        """
        total_char = len(self.augmented_characters)
        widths, heights = [], []
        for augmented_character in self.augmented_characters:
            x, y, w, h = augmented_character["encapsulate_bbox"]
            widths.append(w)
            heights.append(h)
        width = sum(widths) + self.char_padding * (total_char - 1)
        height = max(heights)
        return width, height

    def get_cropped_character(self, char_index):
        """
        Get the cropped version of bw_mask & masked on a single character.
        """
        x, y, w, h = self.augmented_characters[char_index]["encapsulate_bbox"]
        bw_mask = self.augmented_characters[char_index]["bw_mask"]
        cropped_bw_mask = bw_mask[y:y + h, x:x + w]
        masked = self.augmented_characters[char_index]["masked"]
        cropped_masked = masked[y:y + h, x:x + w]
        return cropped_bw_mask, cropped_masked

    def get_paired_cropped_characters(self):
        """
        Get list of cropped version of bw_mask & masked on all characters in augmented_characters.
        """
        cropped_bw_masks = []
        cropped_masked_s = []
        for char_index in range(len(self.augmented_characters)):
            cropped_bw_mask, cropped_masked = self.get_cropped_character(char_index)
            cropped_bw_masks.append(cropped_bw_mask)
            cropped_masked_s.append(cropped_masked)
        return cropped_bw_masks, cropped_masked_s

    @staticmethod
    def add_obj(background, img, mask, x, y):
        """
        Add object to background at specified coordinates.
        """
        bg = background.copy()
        h_bg, w_bg = bg.shape[0], bg.shape[1]
        h, w = img.shape[0], img.shape[1]
        mask_boolean = mask[:, :, 0] == 255
        mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
        h_part = h - max(0, y + h - h_bg)
        w_part = w - max(0, x + w - w_bg)
        bg[y:y + h_part, x:x + w_part, :] = bg[y:y + h_part, x:x + w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
        return bg

    def create_text_box(self):
        """
        Output the merged characters to be a single text box containing the characters.
        """
        text_box_w, text_box_h = self.get_text_box_size()
        text_box_image = np.zeros((text_box_h, text_box_w, 3), dtype=np.uint8)
        text_box_masked = text_box_image.copy()
        text_box_bw_mask = text_box_image.copy()
        cropped_bw_masks, cropped_masked_s = self.get_paired_cropped_characters()

        x, y = 0, 0
        for index, (bw_mask, masked) in enumerate(zip(cropped_bw_masks, cropped_masked_s)):
            h, w = bw_mask.shape[0], bw_mask.shape[1]
            bw_mask = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2RGB)
            text_box_masked = self.add_obj(text_box_masked, masked, bw_mask, x, y)
            text_box_bw_mask = self.add_obj(text_box_bw_mask, bw_mask, bw_mask, x, y)
            x = x + w + self.char_padding

        return text_box_masked, text_box_bw_mask



# Example usage
# processor = AugmentedTextBoxProcessor(augmented_characters)
# text_box_masked, text_box_bw_mask = processor.create_text_box()

# # Display the result
# cv2.imshow("Text Box Masked", text_box_masked)
# cv2.imshow("Text Box BW Mask", text_box_bw_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
