## Inferencia: ¿llueve mañana?

Predice si mañana llueve en una ciudad de Australia a partir de los datos meteorológicos de hoy. Usa la regresión logística balanceada del TP, con su umbral elegido en validación.

### Archivos
-`inferencia.py`: carga el modelom preprocesa los datos y predice. No entrena.
-`modelo.joblib`: imputadores, escaladores, modelo y umbral generados en la seccion 9 del tp.
-`requirements.txt`: librerias necesarias para la inferencia, con version.
-`Dockerfile`: receta para construir la imagen.
-`ejemplo.csv`: 20 dias de prueba para probar la inferencia.

### Entrada y salida 

La entrada es un CSV con el mismo formato que `weatherAUS` (las columnas que sobran se ignoran y se copian a la salida). Se aceptan valores faltantes.

La salida es el mismo CSV con dos columnas más:

`prob_lluvia`: probabilidad de que llueva mañana.
`RainTomorrow_pred`: Yes o No, según el umbral del modelo.


### Construir la imagen
Desde la carpeta `docker/`:

```
docker build -t lluvia-australia .
```

### Ejecutar el contenedor

Desde la carpeta `docker/`. La carpeta actual se monta en el contenedor como `/datos`, y los dos argumentos son el CSV de entrada y el de salida.

PowerShell (Windows):
```
docker run --rm -v "${PWD}:/datos" lluvia-australia /datos/ejemplo.csv /datos/predicciones.csv
```
Bash(Linux/macOS):
```
docker run --rm -v "$(pwd)":/datos lluvia-australia /datos/ejemplo.csv /datos/predicciones.csv
```
Al terminar, `predicciones.csv` queda en la carpeta `docker/`.

