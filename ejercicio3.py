# Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import matplotlib 

#########################################################################################################

# En el ejercicio c) ocupé la imagen de la tortuga con el filtro de la mediana ya que a mi parecer era 
# el que mejor se veía, pero al compararla con la imagen referencial de la tortuga de colores no se veía
# igual, por qué pasa esto?

# PD: en c) Y d) imprimí más tortugas de las que debería porque quería compararlas
#es porque esta usando la cuantizacion que se usaria para un imagen con un histograma extendido
#########################################################################################################

# # a) Carga y despliegue de imagenes en escala de grises.

# # Carga de imagenes
img = cv2.imread('tortugas1.jpg',0)
img2 = cv2.imread('tortugas2.jpeg',0)

# Despliegue de imagenes
plt.figure(figsize=(10, 5))  # Ajusta el tamaño de la figura
plt.subplot(1, 2, 1)  # Define que en la pantalla se mostrarán una fila y dos columnas
plt.imshow(img, cmap='gray')  # Muestra la imagen en escala de grises
plt.title('Tortuga 1')  # Títul que se mostrará
plt.axis('off')  # Oculta los ejes

plt.subplot(1, 2, 2)  
plt.imshow(img2, cmap='gray')  
plt.title('Tortuga 2')  
plt.axis('off')  

# Muestra ambas imagenes en una ventana
plt.show()  

# b) Aplique los filtros de media, mediana y gaussiano vistos en clases (con el menor suavizado
# posible para reducir el ruido), a cada imagen. Despliegue los resultados de cada imagen en
# escala de grises y muestre los resultados en subplots con los nombre correspondientes. Elija
# el que considere mejor para cada imagen y explique la selección de cada filtro.

# Aplica los filtros con el menor suavizado visto en clases
img_media = cv2.blur(img, (5,5)) 
img_mediana = cv2.medianBlur(img, 5)
img_gauss = cv2.GaussianBlur(img, (5,5), 0)

img2_media = cv2.blur(img2, (5,5))
img2_mediana = cv2.medianBlur(img2, 5)
img2_gauss = cv2.GaussianBlur(img2, (5,5), 0)

# Configuración de la figura de Matplotlib para mostrar las imágenes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Mostrar imágenes de la primera tortuga
axs[0, 0].imshow(img_media, cmap='gray')
axs[0, 0].set_title('Media Tortuga 1')
axs[0, 1].imshow(img_mediana, cmap='gray')
axs[0, 1].set_title('Mediana Tortuga 1')
axs[0, 2].imshow(img_gauss, cmap='gray')
axs[0, 2].set_title('Gaussiano Tortuga 1')

# Mostrar imágenes de la segunda tortuga
axs[1, 0].imshow(img2_media, cmap='gray')
axs[1, 0].set_title('Media Tortuga 2')
axs[1, 1].imshow(img2_mediana, cmap='gray')
axs[1, 1].set_title('Mediana Tortuga 2')
axs[1, 2].imshow(img2_gauss, cmap='gray')
axs[1, 2].set_title('Gaussiano Tortuga 2')

# Ocultar ejes
for ax in axs.flat:
    ax.axis('off')

# Mostrar las tortugas
plt.show()

# En teoría, se esperaría que el filtro gaussiano fuera la mejor opción para la imagen 1, dado que 
# ese filtro está diseñado para mejorar el tipo de ruido gaussiano. Sin embargo, en la práctica, el filtro de mediana 
# resultó ser el mejor para ambas imágenes. A pesar de que el filtro de mediana es particularmente adecuado
# para el ruido de sal y pimienta presente en la tortuga 2, ha funcionado muy bien para reducir el ruido en la tortuga 1. 
# Esto se debe a que el filtro de mediana elimina el ruido sin difuminar los bordes.


# c) Cree dos matrices de unos, con dimensiones de 3x3 y 5x5, respectivamente. Luego, mediante
# convolución, aplíquelas por separado sobre las imágenes originales en escala de grises. Mues-
# tre los resultados en subplots. Comente los resultados obtenidos. 

# Se crea matriz de unos de 3x3
size_kernel = 3
kernel = np.ones((size_kernel,size_kernel), dtype = np.float64)
# Se crea matriz de unos de 5x5
size_kernel2 = 5
kernel2 = np.ones((size_kernel2,size_kernel2), dtype = np.float64)

# Se realiza el filtro de la convolucion con el comando signal.convolve2d y se convolucionan las imagenes con la matrices creadas
conv1 = signal.convolve2d(img, kernel, boundary='symm', mode='same') 
conv2 = signal.convolve2d(img, kernel2, boundary='symm', mode='same') 
conv3 = signal.convolve2d(img2, kernel, boundary='symm', mode='same') 
conv4 = signal.convolve2d(img2, kernel2, boundary='symm', mode='same') 

# Configuración de la figura de Matplotlib para mostrar las imágenes
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Mostrar imágenes de la primera tortuga
axs[0, 0].imshow(conv1, cmap='gray')
axs[0, 0].set_title('Convolución 3x3 Tortuga 1')
axs[0, 1].imshow(conv2, cmap='gray')
axs[0, 1].set_title('Convolución 5x5 Tortuga 1')

# Mostrar imágenes de la segunda tortuga
axs[1, 0].imshow(conv3, cmap='gray')
axs[1, 0].set_title('Convolución 3x3 Tortuga 2')
axs[1, 1].imshow(conv4, cmap='gray')
axs[1, 1].set_title('Convolución 5x5 Tortuga 2')

# Ocultar ejes
for ax in axs.flat:
    ax.axis('off')

# Mostrar las tortugas
plt.show()

# ¿Qué efecto realizaron las máscaras de convolución sobre el ruido de las imágenes?
# Las máscaras de convolución suavizan y eliminan el ruido de las imagenes, dependiendo del kernel que se utilice 
# se puede eliminar más o menos ruido, en este caso un kernel de 5x5 elimina más ruido que uno de 3x3.

# d) Cuantifique la imagen que considere mejor filtrada, obtenida de los incisos b) o c), con 16
# niveles de gris y despliegue el resultado en una figura

# Definimos cantidad de niveles para la cuantificación
lvls = 16

# Calculamos la cantidad que representa cada nivel de cuantificación
# Dado que queremos 16 niveles, dividimos el rango total (255) por 15 (16 - 1)
intervalo = 255 / (lvls - 1)

# Cuantificamos la imagen dividiendo por el intervalo y luego multiplicamos por el intervalo
# Esto efectivamente "redondea" cada valor de píxel al nivel de cuantificación más cercano
tortuga_cuant = np.floor(img_mediana / intervalo) * intervalo

# Convertimos a tipo uint8 para visualización
tortuga_cuant = tortuga_cuant.astype('uint8')

# Desplegamos la imagen cuantificada
plt.imshow(tortuga_cuant, cmap='gray')
plt.title('Tortuga cuantificada')
plt.axis('off')
plt.show()

# # código que deberías usar para una imagen con un histograma ya extendido:
# # Definimos cantidad de niveles para la cuantificación
# lvls = 16

# # Calculamos la cantidad que representa cada nivel de cuantificación
# intervalo = 255 / (lvls - 1)

# # Cuantificamos la imagen directamente sin estirar el histograma
# tortuga_cuant = np.floor(img_mediana / intervalo) * intervalo

# # Convertimos a tipo uint8 para visualización
# tortuga_cuant = tortuga_cuant.astype('uint8')

# # Desplegamos la imagen cuantificada
# plt.imshow(tortuga_cuant, cmap='gray')
# plt.title('Tortuga cuantificada')
# plt.axis('off')
# plt.show()


# Si estos valores están cerca de 0 y 255, entonces el histograma ya está extendido.
# Si no es así, y hay una gran cantidad de valores concentrados en un rango más estrecho,
# entonces el histograma no está extendido y esto podría estar causando el problema.


# e) Cree y despliegue una imagen de 3 canales a partir de la imagen cuantificada en el inciso
# anterior con los colores del mapa de colores ‘cool’ de la biblioteca matplotlib. Para obtener el
# mapa de colores utilice la función get cmap disponible en la biblioteca matplotlib. Asigne los
# valores RGB del mapa a los píxeles que comparten el mismo nivel de cuantificación.

# Obtén el mapa de colores 'cool' de Matplotlib
cool_cmap = plt.cm.get_cmap('cool')

# # Asegúrate de que la imagen cuantificada está en el rango [0, 1]
# normalized_tortuga = tortuga_cuant / 255.0

# Aplica el mapa de colores a la imagen normalizada
tortuga_colormap = cool_cmap(tortuga_cuant)

# El mapa de colores incluye un canal alfa, así que lo quitamos para tener sólo RGB
tortuga_rgb = tortuga_colormap[:, :, :-1]

# # Asegúrate de que la imagen coloreada está en el rango [0, 255] y es de tipo uint8
# tortuga_rgb_uint8 = (tortuga_rgb * 255).astype('uint8')

# Muestra la imagen
plt.imshow(tortuga_rgb)
plt.axis('off')  # Oculta los ejes
plt.show()

# f) Aplicando umbralización adaptativa a la imagen filtrada, genere una imagen donde solo se muestre el contorno de la tortuga y desplieguela

# Aplicamos umbralización adaptativa a la imagen filtrada
img_umbral = cv2.adaptiveThreshold(img_mediana, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 9) 

# Desplegamos la imagen umbralizada
plt.imshow(img_umbral, cmap='gray')
plt.title('Imagen umbralizada')
plt.axis('off')
plt.show()
