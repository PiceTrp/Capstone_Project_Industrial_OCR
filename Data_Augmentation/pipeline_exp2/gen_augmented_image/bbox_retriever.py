import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

class BoundingBoxProcessor:
    def __init__(self, result_mask):
        self.result_mask = result_mask # 2d
        self.bounding_boxes = self.find_object_bounding_boxes()


    def find_object_bounding_boxes(self):
        # Find contours in the binary image
        contours, _ = cv2.findContours(self.result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store bounding box coordinates
        bounding_boxes = []

        # Iterate through each contour
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate top-left and bottom-right coordinates
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            # Add the bounding box coordinates to the list
            bounding_boxes.append((top_left, bottom_right))

        return bounding_boxes
    

    def plot_rectangles_on_image(self, save_path=None):
        # Create a copy of the image to draw rectangles on - required 3d
        image_with_rectangles = np.stack([self.result_mask, self.result_mask, self.result_mask], axis=2)

        # Iterate through bounding boxes and draw rectangles
        for top_left, bottom_right in self.bounding_boxes:
            cv2.rectangle(image_with_rectangles, top_left, bottom_right, (255, 0, 0), 2)  # (255, 0, 0) is the color (blue), 2 is the thickness

            # Draw circles at the top-left and bottom-right corners
            cv2.circle(image_with_rectangles, top_left, 5, (0, 255, 0), -1)  # (0, 255, 0) is the color (green), -1 means filled circle
            cv2.circle(image_with_rectangles, bottom_right, 5, (0, 255, 0), -1)

            # Add text labels for the top-left and bottom-right coordinates
            cv2.putText(image_with_rectangles, f'{top_left}', (top_left[0] + 10, top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image_with_rectangles, f'{bottom_right}', (bottom_right[0] + 10, bottom_right[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Plot the image with rectangles
        plt.imshow(image_with_rectangles, cmap='gray')  # Assuming grayscale image
        plt.title('Image with Rectangles')
        # plt.axis('off')  # Turn off axis
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure with high quality
        plt.show()


    def save_bounding_boxes_to_json(self, output_path, text_value):
        # Create a list of dictionaries with top-left and bottom-right coordinates
        boxes_dict = [{"top_left": top_left, "bottom_right": bottom_right} for top_left, bottom_right in self.bounding_boxes]

        # Create the JSON object
        json_data = {"bounding_boxes": boxes_dict, "text": text_value}

        # Save to a JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)

