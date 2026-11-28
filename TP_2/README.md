# TP2 Clasificación: predicción de lluvia en Australia

Aprendizaje Automático 1, Tecnicatura en Inteligencia Artificial (FCEIA, UNR)

Integrantes: Aguirre, Gambande, Paladini

Notebook: [`TP-clasificacion-AA1.ipynb`](TP-clasificacion-AA1.ipynb)
Datos: [`data/weatherAUS_2026C2.csv`](data/weatherAUS_2026C2.csv)
Docker: [`docker/`](docker/readme.md)

## De qué se trata

Predecir si mañana llueve en Australia a partir de las mediciones de un día en una ciudad.
Consigna completa en el PDF de la cátedra.

## Qué hicimos

Analizamos las variables, los faltantes y los outliers. Sacamos `RainfallTomorrow` porque es la
lluvia del día siguiente (fuga de datos). Dividimos por fecha en entrenamiento, validación y
prueba, y imputamos y escalamos solo con entrenamiento. Codificamos el viento con seno y coseno
y la ciudad con latitud y longitud. Entrenamos una regresión logística balanceada, la comparamos
con un modelo base, optimizamos los hiperparámetros, la explicamos con SHAP y probamos PyCaret.
Después entrenamos una red neuronal con TensorFlow, la optimizamos con Optuna, la explicamos con
SHAP, comparamos todos los modelos y desplegamos el elegido con Docker.

## Resultado

F1 de la clase «llueve» en prueba:

| Modelo | F1 |
|---|---|
| Red neuronal optimizada | 0,654 |
| Red neuronal inicial (desplegada) | 0,653 |
| LightGBM (PyCaret) | 0,624 |
| Regresión logística balanceada | 0,615 |
| Modelo base | 0 |

Elegimos la red inicial porque rinde casi igual que la optimizada con muchos menos parámetros.
Es la que está en la carpeta [`docker/`](docker/readme.md).

## Cómo correrlo

El notebook se corrió con Python 3.11. Hay que usar las mismas versiones de scikit-learn y
TensorFlow, porque la consigna 9 vuelve a generar los archivos de `docker/` y tienen que
coincidir con las de [`docker/requirements.txt`](docker/requirements.txt). Entorno nuevo con
las versiones exactas:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install pandas==3.0.6 numpy==2.4.6 scikit-learn==1.5.2 tensorflow==2.21.0 optuna==5.1.0 shap==0.51.0 matplotlib seaborn ipykernel
```

En Windows el entorno se activa con `.venv\Scripts\activate`.

Las celdas de PyCaret (consigna 6) se corrieron en un entorno aparte con Python 3.11
([`requirements-pycaret.txt`](../requirements-pycaret.txt)), porque PyCaret 3.3 necesita una
versión de pandas más vieja que la del notebook.
