# Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video
Este repositorio contiene el código utilizado para el trabajo de memoria de titulación "Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video". 

Las redes neuronales utilizadas corresponden a "Frame-Recurrent Video Super-Resolution" (FRVSR) [1] y "Temporally Coherent GAN for Video Super-Resolution" (TecoGAN) [2].

<p align="center">
  <img height="300" src="images/FRVSR.png" alt="Frame">
</p>

Para realizar cualquier acción es necesario un archivo de configuración `.yaml`. Actualmente hay dos versiones de este archivo de configuración, FRVSR y TecoGAN, cada uno con sus repectivos parámetros de inicialización. 

**Ejemplo:**

`python3 main.py --yaml_file FRVSR.yaml`

Es posible habilitar las siguientes etapas, cada una realizando ciertas acciones con las redes neuronales:
* __Entrenamiento__: entrenamiento de la red neuronal seleccionada.
* __Evaluación__: obtención del video procesado junto con un archivo `.txt` conteniendo el valor de las métricas calculadas.
* __Ejecución__: procesamiento en vivo del video mostrando en pantalla los cuadros por segundo junto a las métricas solicitadas.

## Métrica YOLO

La métrica YOLO corresponde a un sistema alternativo para comparar las imágenes generadas por las redes neuronales. Está basado en la red neuronal "You Only Look Once v3" [3]

<p align="center">
  <img height="300" src="images/sr yolo.png">
</p>

## Bibliografía

[1] [Frame/Recurrent Video Super-Resolution](https://arxiv.org/pdf/1801.04590.pdf)  
[2] [Temporally Coherent GAN fo Video Super-Resolution](https://arxiv.org/pdf/1811.09393.pdf)  
[3] [You Only Look Once v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)  
