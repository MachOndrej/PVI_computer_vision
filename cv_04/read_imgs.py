import os
import cv2

# Directory containing your images
folder_path = 'PVI_CV03'

# Create an empty list to store the images
image_list = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # Check for image file extensions
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        if image is not None:
            # Append the image to the list
            image_list.append(image)

# Now, image_list contains all the images from the folder
# You can access individual images using indexing, e.g., image_list[0]
