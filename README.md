# Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video
Este repositorio contiene el código utilizado para el trabajo de memoria de titulación "Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video".

La súper-resolución es el conjunto de técnicas para aumentar la resolución espacial de una imagen. Métodos tales como la interpolación lineal o bicúbica solo le dan cierto valor a la información faltante mediante la ponderación de los pixeles adyacentes y que la imagen tenga cierta coherencia a nivel de color, mientras que la súper-resolución busca generar detalles acordes aproximando el resultado a la imagen correspondiente en alta resolución.

<p align="center">
  <img height="300" src="/images/super resolution.png">  
</p>

Las redes neuronales utilizadas corresponden a "Frame-Recurrent Video Super-Resolution" (FRVSR) [1] y "Temporally Coherent GAN for Video Super-Resolution" (TecoGAN) [2].

Para realizar cualquier acción es necesario un archivo de configuración `.yaml`. Actualmente hay dos versiones de este archivo de configuración, FRVSR y TecoGAN, cada uno con sus repectivos parámetros de inicialización. 

**Ejemplo:**

`python3 main.py --yaml_file FRVSR.yaml`

Las etapas habilitadas en el programa son las siguientes:
* __Entrenamiento__: entrenamiento de la red neuronal seleccionada.
* __Evaluación__: obtención del video procesado junto con un archivo `.txt` conteniendo el valor de las métricas calculadas.
* __Ejecución__: procesamiento en vivo del video mostrando en pantalla los cuadros por segundo junto a las métricas solicitadas.

## Métrica YOLO

La métrica YOLO corresponde a un sistema alternativo para comparar las imágenes generadas por las redes neuronales. Está basado en la red neuronal "You Only Look Once v3" [3], un sistema de detección de objetos en el estado del arte. 

<p align="center">
  <img height="400" src="images/yolo example.png">
</p>

Este método se encuentra implementado en la etapa de __evaluación__, en donde la idea es obtener el video en alta resolución a partir del modelo deseado para luego, a través de la red YOLO, obtener una lista de los obtejos detectados junto con su porcentaje de certeza.

<p align="center">
  <img height="300" src="images/sr yolo.png">
</p>

## Bibliografía

[1] [Frame/Recurrent Video Super-Resolution](https://arxiv.org/pdf/1801.04590.pdf)  
[2] [Temporally Coherent GAN fo Video Super-Resolution](https://arxiv.org/pdf/1811.09393.pdf)  
[3] [You Only Look Once v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)  
