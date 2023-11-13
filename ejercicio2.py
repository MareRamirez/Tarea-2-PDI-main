# Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# Parte 2.A

img = cv2.imread('eye.jpg') # Se lee la imagen
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Se convierte la imagen a escala de grises

# Dividimos la imagen en dos partes verticalmente para poder aplicar 
# de mejor forma el umbral en cada lado de la imagen, de esta forma 
# se consigue mejor calidad en la imagen resaltada
mitad_ancho = img_gray.shape[1] // 2
lado_izquierdo = img_gray[:, :mitad_ancho]
lado_derecho = img_gray[:, mitad_ancho:]

umbral_izquierdo = 129  # Umbral para el lado izquierdo
umbral_derecho = 161   # Umbral para el lado derecho

# Máscaras binarias para cada lado
mascara_izquierda = lado_izquierdo > umbral_izquierdo
mascara_derecha = lado_derecho > umbral_derecho

# Se hace la conversión a np.uint8 debido a que Matplotlib espera que los valores 
# de la imagen estén en el rango [0, 255] para mostrar correctamente la imagen 
# en escala de grises

mascara_izquierda = np.uint8(mascara_izquierda) 
mascara_derecha = np.uint8(mascara_derecha) 

# Combinación de las máscaras para tener la máscara completa
mascara_completa = np.hstack((mascara_izquierda, mascara_derecha))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Imagen en escala de grises')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mascara_completa, cmap='gray')
plt.title('Máscara')
plt.axis('off')

plt.show()

# Parte 2.B
# Aplicar la mascara obtenida a la imagen original y desplegar la imagen original junto a la
# imagen obtenida, ambas en escala de grises

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Se convierte la imagen a escala de grises
img_gray = np.uint8(img_gray) # Se convierte la imagen a np.uint8
img_gray = img_gray * mascara_completa # Se aplica la mascara a la imagen
img_gray = np.uint8(img_gray) # Se convierte la imagen a np.uint8

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Imagen con máscara')
plt.axis('off')

plt.show()

# Parte 2.C 
# Aplicar los filtros de Roberts, Sobel y Prewitt a la imagen obtenida en el punto anterior 

# Se aplica el filtro de Roberts, Sobel y Prewitt a la imagen obtenida en el punto anterior
roberts = filters.roberts(img_gray)
prewitt = filters.prewitt(img_gray)
sobel = filters.sobel(img_gray)

# Se muestran las imagenes con los filtros aplicados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(np.abs(roberts), cmap='gray')
plt.title('Roberts con máscara')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(sobel), cmap='gray')
plt.title('Sobel con máscara')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(prewitt), cmap='gray')
plt.title('Prewitt con máscara')
plt.axis('off')

plt.show()

# Parte 2.D
# Mostrar los bordes sobre la imagen original. Para ello debe crear una imagen RGB donde los
# bordes queden resaltados con tonos amarillos y la imagen original aparezca en tonalidades
# verdes. Escoja la imagen de bordes obtenida en 2.c) que considere mejor. Recuerde que para
# realizar esto, es necesario trabajar cada canal por separado. 

# Convierte la imagen original a tonos de verde
img_verde = img.copy()
img_verde[:, :, 0] = 0  
img_verde[:, :, 2] = 0  

# Convertir la imagen a escala de grises
img_gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mezclar la imagen en escala de grises con el canal verde
img_verde_mezclado = cv2.addWeighted(img_verde, 0.5, cv2.merge([img_gray_scale, img_gray_scale, img_gray_scale]), 0.5, 0)

roberts = filters.roberts(img_gray)

# Se normalizar y umbralizar la imagen de bordes Roberts
roberts_normalizado = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)
_, mask_bordes = cv2.threshold(roberts_normalizado, 25, 255, cv2.THRESH_BINARY)  

# Convertir la imagen original a tonos de verde
img_verde_mezclado = cv2.addWeighted(img_verde, 0.5, cv2.merge([img_gray_scale, img_gray_scale, img_gray_scale]), 0.5, 0)

# Aplicar bordes amarillos en la imagen 
img_rgb = img_verde_mezclado.copy()
img_rgb[mask_bordes > 0] = [0, 165, 255]  

# Mostrar la imagen 
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title('Imagen con bordes de Roberts')
plt.axis('off')
plt.show()

# Parte 2.E
# Dada la imagen original cargada en el paso a) (“eye.jpg”) obtenga una imagen del ojo sin los
# vasos sanguineos de este. Para lograr este resultado, se puede apoyar en la mascara de las venas
# y arterias obtenida en el paso a). Luego, a la imagen resultante, aplicar un filtro de suavizado
# para desenfocarlo y despliegue los resultados. Utilizar como fondo la imagen desenfocada que
# obtuvo en el paso anterior. Finalmente, debe lograr que las venas y las arterias se resalten
# con color verde y que el fondo difuminado sea de color azul. Puede realizar esto asignando
# la mascara y el fondo por separado a los canales de una imagen RGB utilizando el comando
# np.dstack(R,G,B), para unificar distintos canales en una sola imagen.

# Crear la máscara invertida
# mascara_invertida = 255 - mascara_completa


