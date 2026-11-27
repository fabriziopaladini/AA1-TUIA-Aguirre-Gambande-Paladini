## Inferencia: ¿llueve mañana?

Predice si mañana llueve en una ciudad de Australia a partir de los datos meteorológicos de hoy. Usa la red neuronal inicial del TP (la elegida en la consigna 8), con su umbral elegido en validación.

### Archivos

- `inferencia.py`: carga el preprocesamiento y la red, preprocesa los datos y predice. No entrena.
- `modelo.joblib`: imputadores, escaladores, tablas fijas de codificación, columnas de entrada de la red y umbral, generados en la consigna 9 del notebook.
- `red.keras`: la red neuronal entrenada, en el formato propio de Keras.
- `requirements.txt`: librerías necesarias para la inferencia, con versión. Son las mismas versiones con las que se entrenó, porque `modelo.joblib` y `red.keras` se tienen que cargar con ellas.
- `Dockerfile`: receta para construir la imagen.
- `ejemplo.csv`: 20 días del período de prueba para probar la inferencia.

### Entrada y salida

La entrada es un CSV con el mismo formato que `weatherAUS`. Las columnas que sobran se ignoran y se copian a la salida. Se aceptan valores faltantes, que se imputan igual que en el notebook. Si falta una columna, o aparece una ciudad o una dirección de viento que no estaba en el entrenamiento, el script termina con un error que dice cuál es.

La salida es el mismo CSV con dos columnas más:

- `prob_lluvia`: probabilidad de que llueva mañana.
- `RainTomorrow_pred`: Yes o No, según el umbral del modelo.

### Construir la imagen

Desde la carpeta `docker/`:

```
docker build -t lluvia-australia .
```

La primera vez tarda unos minutos, porque instala TensorFlow.

### Ejecutar el contenedor

Desde la carpeta `docker/`. La carpeta actual se monta en el contenedor como `/datos`, y los dos argumentos son el CSV de entrada y el de salida.

PowerShell (Windows):

```
docker run --rm -v "${PWD}:/datos" lluvia-australia /datos/ejemplo.csv /datos/predicciones.csv
```

Bash (Linux/macOS):

```
docker run --rm -v "$(pwd)":/datos lluvia-australia /datos/ejemplo.csv /datos/predicciones.csv
```

Al terminar, `predicciones.csv` queda en la carpeta `docker/`.
