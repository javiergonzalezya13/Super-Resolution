# Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video
Este repositorio contiene el código utilizado para el trabajo de memoria de titulación "Evaluación de Redes Neuronales Artificiales Aplicadas a Súper-Resolución en Video" y su uso.

La súper-resolución es el conjunto de técnicas para aumentar la resolución espacial de una imagen. Métodos tales como la interpolación lineal o bicúbica solo le dan cierto valor a la información faltante mediante la ponderación de los pixeles adyacentes y que la imagen tenga cierta coherencia a nivel de color, mientras que la súper-resolución busca generar detalles acordes aproximando el resultado a la imagen correspondiente en alta resolución.

<p align="center">
  <img height="400" src="/images/super resolution.png">  
</p>

Las redes neuronales utilizadas corresponden a "Frame-Recurrent Video Super-Resolution" (FRVSR) [1] y "Temporally Coherent GAN for Video Super-Resolution" (TecoGAN) [2].

## Funcionamiento
Las etapas habilitadas en la ejecución del programa son las siguientes:
* __Entrenamiento__: Entrenamiento de la red neuronal seleccionada bajo los parámetros definidos.
* __Evaluación__: Procesamiento de uno o varios videos, lo cual da como resultado el  archivo `metrics.txt` que contiene las diferentes métricas calculadas.
* __Ejecución simple__: Procesamiento en vivo de un video o por medio de la cámara, mostrando en pantalla el resultado de  de aplicar súper-resolución  por segundo junto a las métricas solicitadas.

Para realizar cualquier acción es necesario un archivo de configuración `.yaml` como único parámetro de entrada y el cual se indica al momento de la ejecución del programa. 

**Ejemplo:**
`python3 main.py --yaml_file FRVSR.yaml`

El archivo `.yaml` posee la misma estructura tanto para FRVSR como TecoGAN, excepto por un parámetro exclusivo de TecoGAN que, en caso de que existiese, indica el archivo con los valores de los parámetros internos de su discriminador.  El siguiente ejemplo contiene todos los parámetros que se pueden configurar, seguido de la descripción correspondiente de cada uno. 

**Ejemplo:**
```
root_dir: ./

stage:
	train: true
	eval: true
	run: true
	rec: false

data:
	low_res: 64
	high_res: 256
	channels: 3
	upscale: 4
	videos:  [./dataset/cars, ./dataset/airplanes]
	rows: 2
	cols: 2

cnn:
	model: TecoGAN
	pretrained_model: model_weights_250000.h5

train:
	iterations: 950000
	c_frames: 10
	batch_size: 4
	pretrained_disc: disc_model_weights_250000.h5
	sample_freq: 2500
	checkpoint_freq: 10000
	info_freq: 1000

eval:
	watch: True
	output_dir: ./cars_airplanes_metrics/

run:
	video: volvo_car.mp4
```

Al ejecutar el programa, `root_dir` indica el directorio a partir de la cual se buscan los datos a procesar, modelos pre-entrenados y se crean directorios para almacenar los datos de salida. 

La red neuronal a utilizar está dada por `model`, en donde sus posibles valores `frvsr` y `tecogan`, indican si se debe hacer uso de la red FRVSR o TecoGAN respectivamente. Ya sea para el entrenamiento, evaluación o ejecución simple, `pretrained_model` corresponde al archivo que contiene el valor de los pesos a cargar en la red neuronal.

La baja y alta resolución para las redes neuronales, tanto en el alto como el ancho de los píxeles, está dada por `low_res` y `high_res` respectivamente, mientras que el número de canales en ambos casos se indica por medio de `channels`. Se advierte que si la proporción entre `low_res` y `high_res` no es igual a `upscale`, el programa indica un error con respecto a esto y termina inmediatamente.

Ya sea en el entrenamiento o evaluación, `videos` corresponde al directorio o directorios que contienen los archivos de video a utilizar. En caso de que ambos procesos se lleven a cabo en la misma ejecución, el resultado de la evaluación correspondería a las métricas calculadas para los datos de entrenamiento.

En el caso de la evaluación y ejecución simple, el producto entre `rows` y `cols` determina la cantidad de sub imágenes a procesar, en donde estás sub imágenes son el resultado de dividir la imagen original de interés en cuadrantes. Esto permite que la red neuronal sea capaz de procesar imágenes de mayor tamaño, sin tener que modificar la configuración de la imagen de entrada y salida. 

| Parámetro | Descripción                    |
| ------------- | ------------------------------ |
|`root_dir`| Directorio principal en donde se ejecuta el programa.|
|`train`| Habilita la etapa de entrenamiento.|
|`eval`|Habilita la etapa de evaluación|
|`run`| Habilita la etapa de ejecución simple.|
|`low_res`| Baja resolución.|
|`high_res`|Alta resolución.|
|`channels`|Número de canales de imagen.|
|`upscale`|Escalamiento a realizar para llevar la baja resolución a alta resolución.|
|`videos`|Carpeta de videos o lista de carpeta de videos para entrenar y/o evaluar|
|`rows`|Cantidad de filas a expandir en la imagen final.|
|`cols`|Cantidad de columnas a expandir en la imagen final.|
|`model`|Determina el modelo de red neuronal a utilizar.|
|`pretrained_model`|Archivo con los pesos del modelo pre-entrenado|





### Entrenamiento

Esta etapa se lleva a cabo hasta alcanzar el total de iteraciones indicadas por `iterations`.

Al comienzo de cada iteración, se obtiene a partir de los videos un número de muestras indicadas por `batch_size`, donde cada muestra está compuesta por una cantidad consecutiva de imágenes dada por `c_frames`. 

`checkpoint_freq` determina cada cuántas iteraciones se debe guardar el valor de los pesos actuales y una actualización del archivo `.yaml`, la cual permite una reanudación fácil en caso de que el entrenamiento sea interrumpido. En caso de ocurrir esto, es a partir del nombre del archivo `pretrained_model` que se determina la iteración a partir de la cual se debe continuar entrenando.

Para la red TecoGAN,  se aclara que el archivo `pretrained_model` contiene los valores de los pesos correspondientes al generador, mientras que los del discriminador están dados por `pretrained_disc`.

`sample_freq` indica cada cuántas iteraciones se generan y guardan imágenes de muestra. Esto permite al usuario poder supervisar visualmente el progreso del entrenamiento. Por otro lado, `info_freq` da a conocer por consola las métricas calculadas para algunas muestras de la iteración actual, mostrando así su progreso de forma cuantitativa.

| Parámetro | Descripción                    |
| ------------- | ------------------------------ |
| `iterations`      |    Iteraciones totales a realizar.     |
| `c_frames`   |  Cantidad de imágenes consecutivas a extraer de cada muestra.     |
|`batch_size`|Cantidad de muestras a obtener de los videos.|
|`pretrained_model`|Archivo que contiene los pesos del modelo pre-entrenado a utilizar.|
|`pretrained_disc`|Archivo que conteiene los pesos del modelo pre-entrenado del discriminador a utilizar para TecoGAN.|
|`sample_freq`| Cantidad de iteraciones a ocurrir para generar una imagen de muestra a partir de la red neuronal.|
|`checkpoint_freq`|Cantidad de iteraciones a ocurrir para guardar los pesos actuales de la red neuronal.|
|`info_freq`| Cantidad de iteraciones a ocurrir para mostrar por consola las métricas calculadas.|





### Evaluación

La evaluación calcula las métricas de Peak Signal-to-Ratio (PSNR), Structual Similarity Index Measure (SSIM), tiempos de inferencia de la red neuronal y la cantidad de cuadros por segundo, guardando todos estos datos en un archivo `.txt` Adicionalmente, guarda el o los archivos de video generados por las estimaciones de la red neuronal, dando la opción de ver en vivo el resultado del procesamiento al habilitar el parámetro `watch`. Todos los archivos obtenidos durante la evaluación quedan almacenados en el directorio `output_dir`.

| Parámetro | Descripción                    |
| ------------- | ------------------------------ |
|`watch`| Habilita la opción de mostrar los videos en vivo a medida que se procesan.|
|`output_dir`| Directorio de salida.|

### Ejecución simple

La ejecución simple corresponde al procesamiento en vivo del video dado por `video`, en donde se muestra el video original en alta resolución, en baja resolución, el resultado de aplicar interpolación bicúbica a la baja resolución, y, finalmente, el procesamiento por medio de la red neuronal. Se señala que en caso de que un video no sea indicado, se utiliza la cámara por defecto del equipo.

| Parámetros | Descripción                    |
| ------------- | ------------------------------ |
|`video`| Video a procesar. En caso de no especificar este, se utiliza la cámara por defecto.|

## Métrica YOLO

La métrica YOLO corresponde a un sistema alternativo para evaluar las imágenes generadas por las redes neuronales, la cual hace uso de la red neuronal "You Only Look Once v3" [3], un sistema de detección de objetos en el estado del arte.

<p align="center">
  <img height="400" src="images/yolo example.png">
</p>

Este método se encuentra implementado en la etapa de evaluación, en donde la idea es obtener el video en alta resolución a partir del modelo deseado para luego, a través de la red YOLO, obtener una lista de los obtejos detectados junto con su porcentaje de certeza.

<p align="center">
  <img height="300" src="images/sr yolo.png">
</p>

## Bibliografía

[1] [Frame/Recurrent Video Super-Resolution](https://arxiv.org/pdf/1801.04590.pdf)  
[2] [Temporally Coherent GAN fo Video Super-Resolution](https://arxiv.org/pdf/1811.09393.pdf)  
[3] [You Only Look Once v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
