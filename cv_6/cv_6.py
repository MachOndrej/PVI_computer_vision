import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label

image = cv2.imread('pvi_cv06_mince.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
hue_hist = cv2.calcHist([hue], [0], None, [256], [0, 256])      # Histogram
_, binary_image = cv2.threshold(hue, 40, 1, cv2.THRESH_BINARY)  # Segmentace (...to binary)
hue_bin_img = binary_image                                      # Save to other variable

"""Define Figure1"""
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
# 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(rgb)
plt.title('Original Image')
# 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(hue, cmap='jet')
plt.title('Hue Image')
plt.colorbar()
# 3rd position
fig.add_subplot(rows, columns, 3)
plt.plot(hue_hist, linewidth=0.8)
plt.title('Hue Image Histogram')
# 4th position
hue_bin_img = np.ones_like(hue_bin_img) - hue_bin_img   # invert values
fig.add_subplot(rows, columns, 4)
plt.imshow(hue_bin_img, cmap='jet')
plt.title('Image Segmentation')
plt.colorbar()
# Show
plt.show()

# noise removal
kernel = np.ones((3, 3), np.uint8)      # 3x3 kernel
opening = cv2.morphologyEx(hue_bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
# Distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# Sure Foreground
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 1] = 0

"""Define Figure2"""
fig = plt.figure(figsize=(13, 7))
rows = 2
columns = 3
# 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(dist_transform, cmap='jet')
plt.title('Distance Transform', fontsize=10)
# 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(sure_fg, cmap='jet')
plt.title('Sure Foreground', fontsize=10)
# 3rd position
fig.add_subplot(rows, columns, 3)
plt.imshow(unknown, cmap='jet')
plt.title('Unknown', fontsize=10)
plt.colorbar()
# 4th position
fig.add_subplot(rows, columns, 4)
plt.imshow(markers, cmap='jet')
plt.title('Markers', fontsize=10)
plt.colorbar()
# 5th position
# Apply watershed
img = rgb.copy()
markers = cv2.watershed(img, markers)
border = np.zeros_like(markers, dtype=np.uint8)
border[markers == -1] = 1
border = cv2.dilate(border, kernel, iterations=1)   # Dilate
fig.add_subplot(rows, columns, 5)
plt.imshow(border, cmap='jet')
plt.title('Watershed Border', fontsize=10)
plt.colorbar()
# 6th position
# Mark watershed border in binary hue img
watershed_bin_img = hue_bin_img.copy()
watershed_bin_img[border >= 1] = 0
fig.add_subplot(rows, columns, 6)
plt.imshow(watershed_bin_img, cmap='jet')
plt.title('Binary Image with Watershed', fontsize=10)
plt.colorbar()
# Show
plt.show()
"""Define Figure3"""
fig = plt.figure(figsize=(13, 7))
rows = 1
columns = 3
# 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(watershed_bin_img, cmap='jet')
plt.title('Binary Image with Watershed', fontsize=10)
plt.colorbar()
# 2nd position
label_hue, ncc = label(watershed_bin_img)
fig.add_subplot(rows, columns, 2)
plt.imshow(label_hue, cmap='jet')
plt.title('Region Ident.', fontsize=10)
plt.colorbar()
# 3rd position
cleaned_binary_image = watershed_bin_img.copy()
for i in range(1, ncc + 1):
    marked_area = np.where(label_hue == i)
    area_size = len(marked_area[0])
    if area_size < 1000:
        cleaned_binary_image[label_hue == i] = 0
fig.add_subplot(rows, columns, 3)
plt.imshow(cleaned_binary_image, cmap='jet')
plt.title('Result - Binary Image', fontsize=10)
plt.colorbar()
plt.show()


def kernel_construction(n):
    return np.ones((n, n), np.uint8)


def granulometrie(data, sizes=None):
    out = np.zeros_like(data, dtype=np.uint8)
    if sizes is None:
        sizes = range(3, 64 + 1)
    for n in sizes:
        out += cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel_construction(n), iterations=1)
    return out


granulo_sizes = range(3, 65)
granulo = granulometrie(cleaned_binary_image, granulo_sizes)

x_axis = range(40, 65)
granulometric_spectrum = cv2.calcHist([granulo], [0], None, [25], [40, 65])

for i in x_axis:
    object_count = len(granulo[granulo == i])/i**2
    if object_count < 0.9:
        continue
    object_count = int(np.floor(object_count))
    print("No. objects: ", object_count, " size: ", i, " x ", i)

"""Define Figure4"""
fig = plt.figure(figsize=(13, 7))
rows = 1
columns = 2
# 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(granulo, cmap='jet')
plt.title('Result - Granulometry', fontsize=10)
plt.colorbar()
# 2nd position
fig.add_subplot(rows, columns, 2)
plt.plot(x_axis, granulometric_spectrum)
plt.title('Granul. Image Histogram', fontsize=10)
plt.show()
print('done')
