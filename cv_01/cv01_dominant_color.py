# Ukol: detekujte auto, zjistěte barvu, rozpoznejte SPZ
# C) URCENI DOMINANTNI BARVY
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('cropped_img.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Specify the number of clusters (colors) to detect (e.g., 1 for dominant color)
num_clusters = 1

# Create a K-means clustering model and fit it to the pixel data
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get the RGB values of the cluster centers, representing the dominant color(s)
dominant_colors = kmeans.cluster_centers_.astype(int)

# Convert the dominant color(s) from NumPy array to a list
dominant_colors = dominant_colors.tolist()
R, G, B = dominant_colors[0][0:3]
rgb_string = str(R) +', ' + str(G) +', ' + str(B)
# Print the RGB values of the dominant color(s)
print(f"Dominant Color: RGB({dominant_colors[0][0]}, {dominant_colors[0][1]}, {dominant_colors[0][2]})")
# Reprezentace plt
dominant_color_img = np.zeros((100, 100, 3), dtype='uint8')
dominant_color_img[:, :, :] = dominant_colors
plt.title('RGB: '+rgb_string)
plt.imshow(dominant_color_img)
plt.show()

