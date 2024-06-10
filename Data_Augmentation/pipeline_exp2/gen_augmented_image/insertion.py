import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Insertion:
    def __init__(self, background_processor, text_box_processor, overlay_value=20):
        # background initiate
        self.background_image = background_processor.background_image
        self.insertion_mask = background_processor.insertion_mask
        self.placable_topleft, self.placable_bottomright = background_processor.placable_topleft, background_processor.placable_bottomright
        # text box initiate
        self.object_img = text_box_processor.text_box_masked
        self.object_mask = text_box_processor.text_box_bw_mask
        self.overlay_value = overlay_value
        # for plot
        self.plot_images = {"background_image": self.background_image,
                            "insertion_mask": self.insertion_mask,
                            "text_box_masked": self.object_img,
                            "text_box_bw_mask": self.object_mask}


    def implement_insertion(self, visualize=False, verbose=False):
        random_x, random_y = self.get_random_coordinate()
        random_alpha = self.get_random_alpha_value()

        if verbose:
            print(f"Random selected coordinates (x,y) = ({random_x},{random_y})")
            print(f"Alpha value = {random_alpha}")
            print(f"background_image : {self.background_image.shape}")
            print(f"insertion_mask : {self.insertion_mask.shape}")
            print(f"object_img: {self.object_img.shape}")
            print(f"object_mask: {self.object_mask.shape}")

        # get insert result of text_box_image, text_box_mask at (x,y). same size as background image
        text_box_image, text_box_mask = self.get_inserted_object_and_mask(random_x, random_y, visualize)

        if verbose:
            print(f"text_box_image: {text_box_image.shape}")
            print(f"text_box_mask: {text_box_mask.shape}")

        # get text_box_image, text_box_mask only part that intersect with background insertion_mask
        intersect_object_img, intersect_object_mask = self.intersect_insert_region(insert_region=self.insertion_mask,
                                                                                    object_img=text_box_image, 
                                                                                    object_mask=text_box_mask, 
                                                                                    visualize=visualize)
        # apply alpha blending - get the result image
        blended = self.alpha_blend(foreground=intersect_object_img, 
                                    background=self.background_image, 
                                    alpha_mask=intersect_object_mask, 
                                    transparency=random_alpha,
                                    visualize=visualize)
        
        # plot
        if visualize:
            self.plot_images['result_image'] = blended
            self.plot_all_steps()

        return blended


    def get_random_coordinate(self):
        """
        random coordinates of placable range which is in area of insertion_mask
        can add overlay_value for out-of-range insertion
        """
        x1,y1 = self.placable_topleft
        x2,y2 = self.placable_bottomright
        h, w = self.object_img.shape[:-1]
        # find range that is able to randomly place the top-left coordinate of text box on the background image
        # plus-minus to let's object missing some part at each corner for augmentation purpose 
        placable_width = (x1, x2-w)
        placable_height = (y1, y2-h)
        random_width_range = (placable_width[0] - self.overlay_value, placable_width[1] + self.overlay_value)
        random_height_range = (placable_height[0] - self.overlay_value, placable_height[1] + self.overlay_value)

        # get random top-left coordinates
        random_x = random.randint(random_width_range[0], random_width_range[1])
        random_y = random.randint(random_height_range[0], random_height_range[1])
        return (random_x, random_y)
    

    def get_random_alpha_value(self):
        return random.uniform(0.3, 0.8)
    

    def _place_object(self, background, object_img, object_mask, x, y):
        # original size of background image & object image (text box image)
        h_bg, w_bg = background.shape[0], background.shape[1]
        h, w = object_img.shape[0], object_img.shape[1]

        # placable pixels
        mask = np.stack([object_mask, object_mask, object_mask], axis=2)
        mask_boolean = mask[:,:,0] == 255 # white
        mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

        # for debugging
        # print(f"mask: {mask.shape}")
        # print(f"mask_boolean: {mask_boolean.shape}")
        # print(f"mask_rgb_boolean: {mask_rgb_boolean.shape}")

        # Add image for the visible part of object image
        if x >= 0 and y >= 0:
            h_part = h - max(0, y + h - h_bg) # h_part - part of the image which overlaps background along y-axis
            w_part = w - max(0, x + w - w_bg) # w_part - part of the image which overlaps background along x-axis
            
            if len(background.shape) == 2:
                background[y:y+h_part, x:x+w_part] = background[y:y+h_part, x:x+w_part] * ~mask_boolean[0:h_part, 0:w_part] + (object_mask * mask_boolean)[0:h_part, 0:w_part]
            if len(background.shape) == 3:
                background[y:y+h_part, x:x+w_part, :] = background[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (object_img * mask_rgb_boolean)[0:h_part, 0:w_part, :]

        elif x < 0 and y < 0:
            h_part = h + y
            w_part = w + x

            if len(background.shape) == 2:
                background[0:0+h_part, 0:0+w_part] = background[0:0+h_part, 0:0+w_part] * ~mask_boolean[h-h_part:h, w-w_part:w] + (object_mask * mask_boolean)[h-h_part:h, w-w_part:w]
            if len(background.shape) == 3:
                background[0:0+h_part, 0:0+w_part, :] = background[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[h-h_part:h, w-w_part:w, :] + (object_img * mask_rgb_boolean)[h-h_part:h, w-w_part:w, :]

        elif x < 0 and y >= 0:
            h_part = h - max(0, y+h-h_bg)
            w_part = w + x

            if len(background.shape) == 2:
                background[y:y+h_part, 0:0+w_part] = background[y:y+h_part, 0:0+w_part] * ~mask_boolean[0:h_part, w-w_part:w] + (object_mask * mask_boolean)[0:h_part, w-w_part:w]
            if len(background.shape) == 3:
                background[y:y+h_part, 0:0+w_part, :] = background[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[0:h_part, w-w_part:w, :] + (object_img * mask_rgb_boolean)[0:h_part, w-w_part:w, :]

        elif x >= 0 and y < 0:
            h_part = h + y
            w_part = w - max(0, x+w-w_bg)

            if len(background.shape) == 2:
                background[0:0+h_part, x:x+w_part] = background[0:0+h_part, x:x+w_part] * ~mask_boolean[h-h_part:h, 0:w_part] + (object_mask * mask_boolean)[h-h_part:h, 0:w_part]
            if len(background.shape) == 3:
                background[0:0+h_part, x:x+w_part, :] = background[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean[h-h_part:h, 0:w_part, :] + (object_img * mask_rgb_boolean)[h-h_part:h, 0:w_part, :]


    def get_inserted_object_and_mask(self, x, y, visualize=False):
        """
        Input: top-left of randomly selected insert position: (x,y)
        Output: inserted_text_box (3d) & inserted_text_box_mask (2d) (same size as background_image)
        Obj: get the exact position of the text box that will be next placed in the background image. 
        The text box's part can be loss due to overlay insertion. The result text box and its mask 
        will be used for alpha blending.
        """
        # set-up inputs
        bg_mask = np.zeros(self.background_image.shape[:2])
        bg_rgb_mask = np.zeros(self.background_image.shape, dtype=np.uint8) # don't forget dtype

        # for debugging
        # print(f"bg_mask: {bg_mask.shape}")
        # print(f"bg_rgb_mask: {bg_rgb_mask.shape}")

        # Place image for the visible part of object image
        self._place_object(bg_mask, self.object_img, self.object_mask, x, y) # for 2d mask - inserted_mask
        self._place_object(bg_rgb_mask, self.object_img, self.object_mask, x, y) # for 3d mask - inserted_object

        # plot
        if visualize:
            self.plot_images['bg_mask'] = bg_mask
            self.plot_images['bg_rgb_mask'] = bg_rgb_mask
            # self.plot_inserted_object_and_mask(self.background_image, self.object_img, self.object_mask, bg_rgb_mask, bg_mask, x, y)
        return bg_rgb_mask, bg_mask
    

    def intersect_insert_region(self, insert_region, object_img, object_mask, visualize=False):
        """
        Intersect the object_img and object_mask with the insert_region.

        Parameters:
        - insert_region: The region where the object can be placed (2D array).
        - object_img: The object image (3D array).
        - object_mask: The object mask (2D array).

        Returns:
        - intersect_object_img: The intersected object image (3D array).
        - intersect_object_mask: The intersected object mask (2D array).
        """

        # Ensure the insert_region and object_mask are single channel and binary
        insert_region = (insert_region > 0).astype(np.uint8)
        object_mask = (object_mask > 0).astype(np.uint8)

        # Combine the masks to get the intersection
        intersect_object_mask = cv2.bitwise_and(insert_region, object_mask) # this is [0,1]
        intersect_object_mask_3ch = np.stack([intersect_object_mask]*3, axis=2)

        # Apply the intersected mask to the object image
        intersect_object_img = cv2.bitwise_and(object_img, object_img, mask=intersect_object_mask)
        intersect_object_mask = intersect_object_mask * 255 # this is [0,255] now

        if visualize:
            self.plot_images['intersect_object_img'] = intersect_object_img
            self.plot_images['intersect_object_mask'] = intersect_object_mask
            # self.plot_intersect(intersect_object_img, intersect_object_mask)
        return intersect_object_img, intersect_object_mask
    

    def alpha_blend(self, foreground, background, alpha_mask, transparency=0.5, visualize=False):
        # Ensure the alpha mask is single channel and convert to float
        if len(alpha_mask.shape) == 3:
            alpha = alpha_mask[:, :, 0]
        else:
            alpha = alpha_mask

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255.0

        # Apply the transparency level
        alpha = alpha * transparency

        # Convert images to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Ensure the alpha mask is 3 channels
        alpha = np.stack([alpha, alpha, alpha], axis=2)

        # Perform alpha blending
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)

        # get result
        blended = cv2.add(foreground, background)
        blended = blended.astype(np.uint8) # !!

        if visualize:
            self.plot_images['foreground'] = foreground
            self.plot_images['background'] = background
            # self.plot_alpha_blending(foreground, background)
        return blended
    

    # ----- for visualization ------
    def plot_all_steps(self):
        num_images = len(self.plot_images)
        num_rows = 4
        num_cols = 3

        positions = [1,4,7,10, 2,5,8,11, 3,6,9,12]
        fig = plt.figure(figsize=(10, 18))

        for i, (title, img) in enumerate(self.plot_images.items()):
            position_index = positions[i]
            ax = fig.add_subplot(num_rows, num_cols, position_index)
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:  # 3d
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust space between plots
        plt.show()


    def plot_inserted_object_and_mask(self, background_image, object_img, object_mask, bg_rgb_mask, bg_mask, x, y):
        plt.imshow(background_image)
        plt.title(f"Background Image (3d)")

        plt.imshow(object_img)
        plt.title(f"Text box image (3d)")

        plt.imshow(object_mask, cmap='gray')
        plt.title(f"Text box binaray mask (2d)")

        # show inserted_object image and mask
        plt.imshow(bg_rgb_mask)
        plt.title(f"inserted_text_box (3d) at specified position")
        
        plt.imshow(bg_mask, cmap='gray')
        plt.title(f"inserted_text_box_mask (2d) at specified position")

        plt.suptitle(f"(x,y) = ({x},{y})", fontsize=16, color='blue')
        plt.tight_layout()
        plt.show()


    def plot_intersect(self, intersect_object_img, intersect_object_mask):
        plt.imshow(intersect_object_img)
        plt.title(f"intersect_object_img")
        plt.show()
        plt.imshow(intersect_object_mask)
        plt.title(f"intersect_object_mask")
        plt.show()


    def plot_alpha_blending(self, foreground, background):
        plt.imshow(foreground.astype(np.uint8))
        plt.title("Foreground with Alpha")
        plt.show()

        plt.imshow(background.astype(np.uint8))
        plt.title("Background with Inverted Alpha")
        plt.show()



# some of error case

# Random selected coordinates (x,y) = (1189,235)
# (x,y) = 1189, 235
# Alpha value = 0.31906960768907017
# bg_mask: (506, 2049)
# bg_rgb_mask: (506, 2049, 3)
# object_img: (212, 849, 3)
# object_mask: (212, 849, 3)
# mask: (212, 849, 3, 3)
# mask_boolean: (212, 849, 3)
# mask_rgb_boolean: (212, 849, 3, 3)
# background_image : (506, 2049)