import cv2
import matplotlib.pyplot as plt
import numpy as np

# Filter Creation
filter_mace = []
for i in range(0,3):
    X0 = np.zeros((4096, 3), dtype=complex)
    for j in range(0,3):
        file_name = 'p' + str(i+1) + str(j + 1) + '.bmp'
        print(file_name)
        im = cv2.imread('PVI_C11/' + file_name)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        vec = np.fft.fft2(im_gray).flatten()
        X0[:,j,] = vec
    X = X0

    Xp = X.conjugate().transpose()  # X^-1
    D = np.zeros((4096, 4096), dtype=complex)

    for k in range(0, 4096):
        d = 0
        for l in range(0, 3):
            d += X[k, l]*X[k, l]
        D[k, k] = (1/3)*d

    Dm1 = np.linalg.inv(D)  # D^-1
    u = np.ones(3)
    big_zavorka = Xp @ Dm1 @ X
    filter_mace.append(Dm1 @ X @ big_zavorka @ u)

# Unknown Image Preparation
unknown = cv2.imread('unknown.bmp')
unknown_rgb = cv2.cvtColor(unknown, cv2.COLOR_BGR2RGB)
unknown_gray = cv2.cvtColor(unknown, cv2.COLOR_BGR2GRAY)
X_unknown = np.fft.fft2(unknown_gray).flatten()

# Results Computation
results = []
for i in range(0, 3):
    R = X_unknown * filter_mace[i]  # element-wise multiply
    results.append(R)

# Reverse the operation using IFFT
restored = []
for ii in range(0, len(results)):
    restored_unknown = np.fft.ifft2(results[ii].reshape(64, 64))
    restored_unknown = np.abs(restored_unknown)
    restored.append(restored_unknown)

for jj in range(0, len(restored)):
    # Calculate the indices for the center of the image
    center_x, center_y = restored[jj].shape[1] // 2, restored[jj].shape[0] // 2
    # Switch the quadrants
    top_left = restored[jj][:center_y, :center_x]
    top_right = restored[jj][:center_y, center_x:]
    bottom_left = restored[jj][center_y:, :center_x]
    bottom_right = restored[jj][center_y:, center_x:]

    # Reassemble the image with switched quadrants
    switched_image = np.vstack((np.hstack((bottom_right, bottom_left)),
                                np.hstack((top_right, top_left))))
    restored[jj] = switched_image

restored_copy = restored.copy()
plt.figure(figsize=(10, 4))
# Plot the first image in the first subplot
plt.subplot(1, 3, 1)
plt.imshow(restored[0], cmap='jet')
plt.colorbar()
# Plot the second image in the second subplot
plt.subplot(1, 3, 2)
plt.imshow(restored[1], cmap='jet')
plt.colorbar()
# Plot the third image in the third subplot
plt.subplot(1, 3, 3)
plt.imshow(restored[2], cmap='jet')
plt.colorbar()
# Adjust layout to prevent clipping of titles
plt.tight_layout()
# Show the figure
plt.show()

# Create a figure and three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
# Define the data for each subplot
titles = ['1', '2', '3']

# Iterate over subplots and plot the surfaces
for ax, restored, title in zip(axes, restored, titles):
    x_data = np.arange(0, 64, 1)
    y_data = np.arange(0, 64, 1)
    X, Y = np.meshgrid(x_data, y_data)
    Z = restored
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
# Adjust layout
plt.tight_layout()
# Show the figure
plt.show()

# Strmost
steepness_maps = {}
for j in range(0, len(restored_copy)):
    O = restored_copy[j][22:42, 22:42].copy()
    O[5:15, 5:15] = 0
    O = O.flatten()
    O = np.delete(O, np.where(O == 0))
    V = restored_copy[j][27:37, 27:37].copy()
    V = V.flatten()
    res_i = (np.max(V) - np.mean(O)) / np.std(O)
    steepness_maps[j + 1] = res_i
# Choose the map with the maximum steepness
max_steepness_key = max(steepness_maps, key=lambda k: np.max(steepness_maps[k]))
max_steepness = steepness_maps[max_steepness_key]

# Final Figure Plot
nearest_file = 'p' + str(max_steepness_key) + str(np.random.randint(1, 4)) + '.bmp'
nearest = cv2.imread('PVI_C11/' + nearest_file)
nearest_rgb = cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 4))
# Plot the first image in the first subplot
plt.subplot(1, 2, 1)
plt.imshow(unknown_rgb)
plt.title('Unknown Im.')
# Plot the second image in the second subplot
plt.subplot(1, 2, 2)
plt.imshow(nearest_rgb)
plt.title('Nearest Class')
plt.tight_layout()
# Show the figure
plt.show()
