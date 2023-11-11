# Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#########################################################################################################

# En el ejercicio c) ocupé la imagen de la tortuga con el filtro de la mediana ya que a mi parecer era 
# el que mejor se veía, pero al compararla con la imagen referencial de la tortuga de colores no se veía
# igual, por qué pasa esto?

# PD: en c) Y d) imprimí más tortugas de las que debería porque quería compararlas
#es porque esta usando la cuantizacion que se usaria para un imagen con un histograma extendido
#########################################################################################################

# # a) Carga y despliegue de imagenes en escala de grises.

# Carga de imagenes
img = cv2.imread('tortugas1.jpg',0)
img2 = cv2.imread('tortugas2.jpeg',0)

# # Despliegue de imagenes
# plt.figure(figsize=(10, 5))  # Ajusta el tamaño de la figura
# plt.subplot(1, 2, 1)  # Define que en la pantalla se mostrarán una fila y dos columnas
# plt.imshow(img, cmap='gray')  # Muestra la imagen en escala de grises
# plt.title('Tortuga 1')  # Títul que se mostrará
# plt.axis('off')  # Oculta los ejes

# plt.subplot(1, 2, 2)  
# plt.imshow(img2, cmap='gray')  
# plt.title('Tortuga 2')  
# plt.axis('off')  

# # Muestra ambas imagenes en una ventana
# plt.show()  

# b) Aplique los filtros de media, mediana y gaussiano vistos en clases (con el menor suavizado
# posible para reducir el ruido), a cada imagen. Despliegue los resultados de cada imagen en
# escala de grises y muestre los resultados en subplots con los nombre correspondientes. Elija
# el que considere mejor para cada imagen y explique la selección de cada filtro.

# # Aplica los filtros con el menor suavizado visto en clases
# img_media = cv2.blur(img, (5,5)) 
# img_mediana = cv2.medianBlur(img, 5)
# img_gauss = cv2.GaussianBlur(img, (5,5), 0)

# img2_media = cv2.blur(img2, (5,5))
# img2_mediana = cv2.medianBlur(img2, 5)
# img2_gauss = cv2.GaussianBlur(img2, (5,5), 0)

# # Configuración de la figura de Matplotlib para mostrar las imágenes
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# # Mostrar imágenes de la primera tortuga
# axs[0, 0].imshow(img_media, cmap='gray')
# axs[0, 0].set_title('Media Tortuga 1')
# axs[0, 1].imshow(img_mediana, cmap='gray')
# axs[0, 1].set_title('Mediana Tortuga 1')
# axs[0, 2].imshow(img_gauss, cmap='gray')
# axs[0, 2].set_title('Gaussiano Tortuga 1')

# # Mostrar imágenes de la segunda tortuga
# axs[1, 0].imshow(img2_media, cmap='gray')
# axs[1, 0].set_title('Media Tortuga 2')
# axs[1, 1].imshow(img2_mediana, cmap='gray')
# axs[1, 1].set_title('Mediana Tortuga 2')
# axs[1, 2].imshow(img2_gauss, cmap='gray')
# axs[1, 2].set_title('Gaussiano Tortuga 2')

# # Ocultar ejes
# for ax in axs.flat:
#     ax.axis('off')

# # Mostrar las tortugas
# plt.show()

# # En teoría, se esperaría que el filtro gaussiano fuera la mejor opción para la imagen 1, dado que 
# # ese filtro está diseñado para mejorar el tipo de ruido gaussiano. Sin embargo, en la práctica, el filtro de mediana 
# # resultó ser el mejor para ambas imágenes. A pesar de que el filtro de mediana es particularmente adecuado
# # para el ruido de sal y pimienta presente en la tortuga 2, ha funcionado muy bien para reducir el ruido en la tortuga 1. 
# # Esto se debe a que el filtro de mediana elimina el ruido sin difuminar los bordes.


# c) Cree dos matrices de unos, con dimensiones de 3x3 y 5x5, respectivamente. Luego, mediante
# convolución, aplíquelas por separado sobre las imágenes originales en escala de grises. Mues-
# tre los resultados en subplots. Comente los resultados obtenidos. 

# Matriz de 1's para convolucionar
size_kernel = 3
kernel = np.ones((size_kernel, size_kernel), dtype=np.float64) 

# ¿Qué efecto realizaron las máscaras de convolución sobre el ruido de las imágenes?
# Las máscaras de convolución suavizan la imagen, eliminando el ruido de la misma. 

# # d) Cuantifique las imagenes , obtenidas de los incisos b) o c), con 16
# # niveles de gris y despliegue el resultado en una figura.

# # Cuantificación de imágenes con 16 niveles de gris
# img_media_16 = np.uint8(img_media / 16) * 16
# img_mediana_16 = np.uint8(img_mediana / 16) * 16
# img_gauss_16 = np.uint8(img_gauss / 16) * 16
# img_conv_3_16 = np.uint8(img_conv_3 / 16) * 16
# img_conv_5_16 = np.uint8(img_conv_5 / 16) * 16

# img2_media_16 = np.uint8(img2_media / 16) * 16
# img2_mediana_16 = np.uint8(img2_mediana / 16) * 16
# img2_gauss_16 = np.uint8(img2_gauss / 16) * 16
# img2_conv_3_16 = np.uint8(img2_conv_3 / 16) * 16
# img2_conv_5_16 = np.uint8(img2_conv_5 / 16) * 16

# # Configura la figura de Matplotlib para mostrar las imágenes en subplots
# fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Ajusta la figura a dos filas y cinco columnas

# # Mostrar imágenes cuantificadas de la primera tortuga
# axs[0, 0].imshow(img_media_16, cmap='gray')
# axs[0, 0].set_title('Media 16 Tortuga 1')
# axs[0, 1].imshow(img_mediana_16, cmap='gray')
# axs[0, 1].set_title('Mediana 16 Tortuga 1')
# axs[0, 2].imshow(img_gauss_16, cmap='gray')
# axs[0, 2].set_title('Gaussiano 16 Tortuga 1')
# axs[0, 3].imshow(img_conv_3_16, cmap='gray')
# axs[0, 3].set_title('Convolución 3x3 16 Tortuga 1')
# axs[0, 4].imshow(img_conv_5_16, cmap='gray')
# axs[0, 4].set_title('Convolución 5x5 16 Tortuga 1')

# # Mostrar imágenes cuantificadas de la segunda tortuga
# axs[1, 0].imshow(img2_media_16, cmap='gray')
# axs[1, 0].set_title('Media 16 Tortuga 2')
# axs[1, 1].imshow(img2_mediana_16, cmap='gray')
# axs[1, 1].set_title('Mediana 16 Tortuga 2')
# axs[1, 2].imshow(img2_gauss_16, cmap='gray')
# axs[1, 2].set_title('Gaussiano 16 Tortuga 2')
# axs[1, 3].imshow(img2_conv_3_16, cmap='gray')
# axs[1, 3].set_title('Convolución 3x3 16 Tortuga 2')
# axs[1, 4].imshow(img2_conv_5_16, cmap='gray')
# axs[1, 4].set_title('Convolución 5x5 16 Tortuga 2')

# # Ocultar los ejes 
# for ax in axs.flat:
#     ax.axis('off')
# # Para ajustar el espacio entre las imagenes
# plt.tight_layout()
# # Mostrar la ventana con todos los subplots
# plt.show()

# # La imagen mejor filtrada es la imagen filtrada con el filtro de mediana, ya que es la que mejor
# # conserva los bordes de la imagen.

# # e) Cree y despliegue una imagen de 3 canales a partir de la imagen cuantificada en el inciso
# # anterior con los colores del mapa de colores ‘cool’ de la biblioteca matplotlib. Para obtener el
# # mapa de colores utilice la función get cmap disponible en la biblioteca matplotlib. Asigne los
# # valores RGB del mapa a los píxeles que comparten el mismo nivel de cuantificación.

# # Asumiendo que `im` es tu objeto de imagen de PIL cargado correctamente
# lvls = 2
# lvls2 = 8

# # Realiza la cuantificación de la imagen a los niveles de colores especificados
# im_quan = im.quantize(lvls)
# im_quan2 = im.quantize(lvls2)

# # Para visualizar las imágenes cuantificadas en una figura de Matplotlib
# import matplotlib.pyplot as plt

# # Configura la figura de Matplotlib para mostrar las imágenes cuantificadas
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Ajusta la figura a una fila y dos columnas

# # Mostrar la imagen cuantificada con 'lvls' colores
# axs[0].imshow(im_quan)
# axs[0].set_title(f'Cuantificación a {lvls} Niveles')

# # Mostrar la imagen cuantificada con 'lvls2' colores
# axs[1].imshow(im_quan2)
# axs[1].set_title(f'Cuantificación a {lvls2} Niveles')

# # Ocultar los ejes para ambas imágenes
# for ax in axs:
#     ax.axis('off')

# # Mostrar la ventana con los subplots
# plt.show()





