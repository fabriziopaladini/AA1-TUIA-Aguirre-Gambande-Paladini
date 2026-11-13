# TP2 Clasificación: predicción de lluvia en Australia

Aprendizaje Automático 1, Tecnicatura en Inteligencia Artificial (FCEIA, UNR)

Integrantes: Aguirre, Gambande, Paladini

Notebook: [`TP-clasificacion-AA1.ipynb`](TP-clasificacion-AA1.ipynb)
Datos: [`data/weatherAUS_2026C2.csv`](data/weatherAUS_2026C2.csv)

## De qué se trata

Predecir si mañana llueve en Australia a partir de las mediciones de un día en una ciudad.
Consigna completa en el PDF de la cátedra.

## Qué hicimos

Analizamos las variables, los faltantes y los outliers. Sacamos `RainfallTomorrow` porque es la
lluvia del día siguiente (fuga de datos). Dividimos por fecha en entrenamiento, validación y
prueba, y imputamos y escalamos solo con entrenamiento. Codificamos el viento con seno y coseno
y la ciudad con latitud y longitud. Entrenamos una regresión logística balanceada, la comparamos
con un modelo base, optimizamos los hiperparámetros, la explicamos con SHAP y probamos PyCaret.

## Resultado

La regresión logística da en prueba un F1 de 0,615 y un AUC de 0,849. El modelo base no detecta
ningún día de lluvia. PyCaret elige LightGBM, que mejora el AUC (0,878) pero detecta menos días
de lluvia.

## Cómo correrlo

La instalación del entorno está en el [README del repo](../README.md). Abrir el notebook y correr
todo de arriba a abajo. Las celdas de PyCaret se corrieron en un entorno aparte con Python 3.11,
porque PyCaret 3.3 no soporta Python 3.12.

## Pendiente

Ítems 7 a 10: red neuronal, comparación de modelos, docker y conclusión.
