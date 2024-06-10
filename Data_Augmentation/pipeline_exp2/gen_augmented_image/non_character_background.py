import cv2
import numpy as np
import albumentations as A

# I have learned that when we call function inside a class, we don't need to pass "self" again

class NonCharacterBackgroundProcessor:
    """
    A class for preprocessing background image to get placable insert area
    for placing generated characters.
    """
    def __init__(self, image_path):
        self.image = cv2.imread(image_path) # original background image
        self.combined_mask = self.get_combined_mask() # placable region
        self.leftmost, self.rightmost = self._find_left_right_most()
        self.cropped_background = self.get_cropped_background()
        self.cropped_combined_mask = self.get_cropped_combined_mask()

        # apply albumentation
        self.transform_position = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.05, p=0.2), # Scale between 95% and 105% of original size
            # A.Rotate(limit=2, p=0.2),               # Rotate between -2 and 2 degrees
            # For translate on x-asix only
            A.ShiftScaleRotate(
                shift_limit_x=0.3,  # Shift up to 30% of the image width along the x-axis
                shift_limit_y=0.0,  # No shift along y-axis
                scale_limit=0,      # No apply, already applied above
                rotate_limit=0,     # No apply, already applied above
                p=0.7,
            ),
        ])
        self.transform_effect = A.Compose([
            A.RandomBrightness(limit=0.1, p=0.3),  # ±10% change in brightness
            A.RandomContrast(limit=0.1, p=0.3),    # ±10% change in contrast
            # A.Blur(blur_limit=5, p=0.2), # blur_limit = kernel = [3, blur_limit]
            # A.GaussNoise(var_limit=(10.0, 50.0), per_channel=False, p=0.2), # per_channel=False = The same noise value on all channels > R, G, B value
        ])

        # final result after all background implementation steps
        self.background_image, self.insertion_mask = self.get_transform_result()
        self.placable_topleft, self.placable_bottomright = self._find_placable_coordinates()


    def _otsu_thresholding(self, gray_image):
        _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold_image


    def _find_largest_contour(self, threshold_image):
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour


    def _get_bounding_rectangle(self, largest_contour):
        """
        This is to get encapsulated rectangle - the rectangle that encapsulates the largest contour
        - y + 20 because we will left space for top border to perform UNION operation with
        the largest contour or else there might remove some part of the roughness region
        of the top metal surface.
        - No transform on y-axis for the background image, so need to keep adding that y
        """
        x, y, w, h = cv2.boundingRect(largest_contour)
        adjusted_y = max(y + 20, 0)
        return x, y, w, h, adjusted_y


    def _create_combined_mask(self, largest_contour):
        # Create a mask for the largest contour as a 2D array
        largest_contour_mask = np.zeros_like(self.image)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # create a mask for the encapsulated rectangle as a 2D array
        x, y, w, h, adjusted_y = self._get_bounding_rectangle(largest_contour)
        encapsulated_rectangle_mask = np.zeros_like(self.image)
        cv2.rectangle(encapsulated_rectangle_mask, (x, adjusted_y), (x + w, y + h), 255, thickness=cv2.FILLED)

        # Combine the masks using bitwise OR
        combined_mask = cv2.bitwise_or(largest_contour_mask, encapsulated_rectangle_mask)
        return combined_mask


    def _find_left_right_most(self):
        # Find the leftmost, rightmost, and topmost non-zero pixels in the combined mask
        leftmost = np.where(self.combined_mask != 0)[1].min()
        rightmost = np.where(self.combined_mask != 0)[1].max()
        return leftmost, rightmost


    def get_combined_mask(self):
        """
        Gets the combined mask and its contours from an image.
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        threshold_image = self._otsu_thresholding(gray_image)

        # Identify the largest contour by area
        largest_contour = self._find_largest_contour(threshold_image)

        # Get combined mask
        combined_mask = self._create_combined_mask(largest_contour)
        return combined_mask


    """
    these 2 functions below are to remove gray color value of left-right corner of the original image
    since we want to get rid of unwanted part resulting from applying perspective transform 
    to get the text image
    """
    def get_cropped_background(self):
        # Crop the original image based on the mask dimensions
        cropped_image = self.image[:, self.leftmost:self.rightmost+1]
        return cropped_image


    def get_cropped_combined_mask(self):
        # Resize the combined mask to match the cropped image size
        cropped_combined_mask = self.combined_mask[:, self.leftmost:self.rightmost+1]
        # To be 2d array as a mask
        cropped_combined_mask = cv2.cvtColor(cropped_combined_mask, cv2.COLOR_BGR2GRAY)
        return cropped_combined_mask
    

    # Albumentation
    def get_transform_result(self):
        # 1st transform - augment position first
        transformed_1 = self.transform_position(image=self.cropped_background, mask=self.cropped_combined_mask)
        transformed_image = transformed_1['image']
        transformed_mask = transformed_1['mask']

        # 2nd transform - augment color value - brightness contrast blur noise - for only background image, don't involved mask
        transformed_2 = self.transform_effect(image=transformed_image)
        transformed_image_2 = transformed_2['image']

        # ensure mask is only 0 and 255
        transformed_mask[transformed_mask > 0] = 255

        # ensure largest contour is the insertion mask
        transformed_mask = self.ensure_largest_contour_mask(transformed_mask)

        return transformed_image_2, transformed_mask # final background_image & insertion_mask
    

    def ensure_largest_contour_mask(self, binary_image):
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a new binary mask for the largest contour
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [largest_contour], 0, (255), -1)
        return mask
    

    def _find_placable_coordinates(self):
        """
        Input: insertion_mask (Non-character Background mask after remove left right)
        Output: top-left & bottom-right of mask insertion area
        """
        # Find the largest contour by sorting contours based on area
        contours, _ = cv2.findContours(self.insertion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the rectangle that encapsulate the largest contour = Mask Insertion Area
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_rect_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

        # get top-left & bottom-right of mask insertion area
        placable_topleft = bounding_rect_contour[0]
        placable_bottomright = bounding_rect_contour[2]
        return placable_topleft, placable_bottomright
