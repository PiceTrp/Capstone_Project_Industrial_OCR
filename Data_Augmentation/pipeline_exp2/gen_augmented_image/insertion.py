import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageOverlay:
    def __init__(self, background_image, text_box_masked, text_box_bw_mask, overlay_value=20):
        self.bg = background_image.copy()
        self.object_img = text_box_masked.copy()
        self.object_img_mask = text_box_bw_mask.copy()
        self.overlay_value = overlay_value

    def get_random_coordinate(self, placable_topleft, placable_bottomright):
        x1,y1 = placable_topleft
        x2,y2 = placable_bottomright
        h, w = self.object_img.shape[:-1]
        # find range that is able to randomly place the top-left coordinate of text box on the background image
        # plus-minus 30 to let's object missing some part for augmentation purpose
        placable_width = (x1, x2-w)
        placable_height = (y1, y2-h)
        random_width_range = (placable_width[0] - overlay_value, placable_width[1] + overlay_value)
        random_height_range = (placable_height[0] - overlay_value, placable_height[1] + overlay_value)

        # get random top-left coordinates
        random_x = random.randint(random_width_range[0], random_width_range[1])
        random_y = random.randint(random_height_range[0], random_height_range[1])

        return (random_x, random_y)

    def get_inserted_object_and_mask(self, x=0, y=0, discard_y_pixels=0, visualize=False):
        # Setup inputs
        bg_mask = np.zeros(self.bg.shape[:2])
        bg_rgb_mask = np.zeros(self.bg.shape, dtype=np.uint8)
        h_bg, w_bg = bg_mask.shape[0], bg_mask.shape[1]
        h, w = self.object_img.shape[0], self.object_img.shape[1]

        mask = np.stack([self.object_img_mask, self.object_img_mask, self.object_img_mask], axis=2)
        mask_boolean = mask[:,:,0] == 255  # white
        mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

        # Calculation for the visible part of object image
        if x >= 0 and y >= 0:
            h_part = h - max(0, y + h - h_bg)
            w_part = w - max(0, x + w - w_bg)
            h_part = max(0, h_part - discard_y_pixels)
            mask_rgb_boolean = mask_rgb_boolean[:h_part, :w_part, :]
            bg_rgb_mask[y:y+h_part, x:x+w_part, :] = (
                bg_rgb_mask[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean + 
                (self.object_img * mask_rgb_boolean)
            )
            bg_mask[y:y+h_part, x:x+w_part] = (
                bg_mask[y:y+h_part, x:x+w_part] * ~mask_boolean[:h_part, :w_part] + 
                (self.object_img_mask * mask_boolean)[:h_part, :w_part]
            )
        elif x < 0 and y < 0:
            h_part = h + y
            w_part = w + x
            mask_rgb_boolean = mask_rgb_boolean[h-h_part:h, w-w_part:w, :]
            bg_rgb_mask[0:0+h_part, 0:0+w_part, :] = (
                bg_rgb_mask[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean + 
                (self.object_img * mask_rgb_boolean)
            )
            bg_mask[0:0+h_part, 0:0+w_part] = (
                bg_mask[0:0+h_part, 0:0+w_part] * ~mask_boolean[h-h_part:h, w-w_part:w] + 
                (self.object_img_mask * mask_boolean)[h-h_part:h, w-w_part:w]
            )
        elif x < 0 and y >= 0:
            h_part = h - max(0, y + h - h_bg)
            w_part = w + x
            mask_rgb_boolean = mask_rgb_boolean[:h_part, w-w_part:w, :]
            bg_rgb_mask[y:y+h_part, 0:0+w_part, :] = (
                bg_rgb_mask[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean + 
                (self.object_img * mask_rgb_boolean)
            )
            bg_mask[y:y+h_part, 0:0+w_part] = (
                bg_mask[y:y+h_part, 0:0+w_part] * ~mask_boolean[:h_part, w-w_part:w] + 
                (self.object_img_mask * mask_boolean)[:h_part, w-w_part:w]
            )
        elif x >= 0 and y < 0:
            h_part = h + y
            w_part = w - max(0, x + w - w_bg)
            mask_rgb_boolean = mask_rgb_boolean[h-h_part:h, :w_part, :]
            bg_rgb_mask[0:0+h_part, x:x+w_part, :] = (
                bg_rgb_mask[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean + 
                (self.object_img * mask_rgb_boolean)
            )
            bg_mask[0:0+h_part, x:x+w_part] = (
                bg_mask[0:0+h_part, x:x+w_part] * ~mask_boolean[h-h_part:h, :w_part] + 
                (self.object_img_mask * mask_boolean)[h-h_part:h, :w_part]
            )

        if visualize:
            plt.imshow(bg_rgb_mask)
            plt.show()

        return bg_rgb_mask, bg_mask