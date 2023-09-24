# Ukol: detekujte auto, zjistěte barvu, rozpoznejte SPZ
# A) DETEKCE AUTA (COLOR BASED)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Nacteni obrazku
image = cv2.imread('cv01_auto.jpg')

# Prevod z BGR do HSV (Hue, Saturation, Value) - udajne dobry napad na pro tento druh uloh
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Horni a dolni threshold pro modrou barvu v HSV - nastaveno rucne :/
lower_blue = np.array([90, 85, 50])
upper_blue = np.array([130, 255, 255])

# tvorba masky pomoci threshold
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
"""
## Snaha trefit co nejpresneji masku pomoci thresholdu
# aplikace masky
blue_car = cv2.bitwise_and(image, image, mask=blue_mask)

# Nahled masky - uzivano pro ladeni thresholds
cv2.imshow('Blue Objects', blue_car)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# tvorba minimální ohraničující obdélník na zaklade masky
bbox = cv2.boundingRect(blue_mask)

img2 = image.copy()     # kopie pro zbytek obrazku
# nakresleni zeleneho obdelniku
x, y, w, h = bbox
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Obrazek s obdelnikem
#cv2.imwrite('cv01_auto_rect.jpg', image)

cv2.imshow('image', image)

# Oriznu si puvodni obrazek (resp. jeho kopii) podle hodnot zeleneho obdelniku
crop_img = img2[y:y+h, x:x+w]

cv2.imshow('cropped', crop_img)
cv2.imwrite('cropped_img.jpg', crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
