import cv2
import numpy as np

# I have learned that when we call function inside a class, we don't need to pass "self" again

class NonCharacterBackgroundProcessor:
    """
    A class for preprocessing background image to get placable insert area
    for placing generated characters.
    """
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.combined_mask = self.get_combined_mask()
        self.leftmost, self.rightmost = self._find_left_right_most()


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
        y + 20 because we will left space for top border to perform UNION operation with
        the largest contour or else there might remove some part of the roughness region
        of the top metal surface.
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