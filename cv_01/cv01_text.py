# Ukol: detekujte auto, zjistěte barvu, rozpoznejte SPZ
# B) DETEKCE TEXTU
import cv2
import easyocr
import matplotlib.pyplot as plt

image = cv2.imread('cropped_img.jpg')
# Nacteni modelu
reader = easyocr.Reader(['en'])
# Vysledky ruzne pripravenych obrazku po zpracovani
result = reader.readtext(image, detail=0)
print('Detected Text (BGR): ', result)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = reader.readtext(image_rgb, detail=0)
print('Detected Text (RGB): ', result)

image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = reader.readtext(image_grey, detail=0)
print('Detected Text (Grey):', result)

denoised_image = cv2.GaussianBlur(image_grey, (5, 5), 0)
result = reader.readtext(denoised_image, detail=0)
print('Detected Text (Denoised):', result)

enhanced_image = cv2.equalizeHist(image_grey)
result = reader.readtext(enhanced_image, detail=0)
# Nahled obrazku
plt.imshow(enhanced_image)
plt.axis('off')
plt.show()
print('Detected Text (Enhanced Contrast):', result, '✓')

print('Real (Correct) Text: 3SC 4898')
