# Margareth Ramirez Valenzuela, Ana María Vargas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import filters

# Cargar la imagen “eye.jpg” y convertirla a escala de grises. Luego, obtener una m ́ascara
# binaria de las venas y arterias. Desplegar la m ́ascara. Si usted desea obtener una m ́ascara
# de mejor calidad, puede dividir la imagen en partes iguales con tal de escoger un umbral de
# binarizaci ́on por zona. Por ejemplo, si dividimos la imagen en 4 partes iguales, es posible
# obtener 4 umbrales, correspondientes a los m ́as precisos para cada zona. Para realizar la
# mascara, puede orientarse a partir de la Figura 4.

# Cargar la imagen
img = cv2.imread('eye.jpg') # Se lee la imagen
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Se convierte a RGB

