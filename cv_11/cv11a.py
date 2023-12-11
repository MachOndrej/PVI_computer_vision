import cv2
import matplotlib.pyplot as plt
import numpy
import os
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
    # alternativa
    #big_zavorka = np.matmul(np.matmul(Xp, Dm1), X)
    filter_mace.append(Dm1 @ X @ big_zavorka @ u)

# Unknown Image Preparation
unknown = cv2.imread('unknown.bmp')
unknown_gray = cv2.cvtColor(unknown, cv2.COLOR_BGR2GRAY)
print(unknown_gray.shape)
X_unknown = np.fft.fft2(unknown_gray).flatten()

# Results Computation
results = []
for i in range(0, 3):
    R = X_unknown * filter_mace[i]  # element-wise multiply
    results.append(R)

# TODO: Zpetna FFT
#A = np.abs(np.fft.ifft2(results[0]))
# Reverse the operation using IFFT
restored_unknown_gray = np.fft.ifft2(results[0].reshape(64, 64)).real

# TODO: Spectra vykreslit

# TODO: Vykreslit puvodni + Unknown img
print('Done')