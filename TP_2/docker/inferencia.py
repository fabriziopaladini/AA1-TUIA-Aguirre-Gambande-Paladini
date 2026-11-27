"""Predice si mañana llueve en una ciudad de Australia.

Lee un CSV con datos meteorológicos crudos (mismo formato que weatherAUS)
y escribe otro con dos columnas más:
    - prob_lluvia: probabilidad de que llueva mañana según la red neuronal.
    - RainTomorrow_pred: "Yes" si prob_lluvia supera el umbral elegido en
      validación, "No" en caso contrario.

Uso:
    python inferencia.py entrada.csv salida.csv
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Oculta los mensajes informativos de TensorFlow al cargar la red
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


CARPETA = Path(__file__).parent
PREPROCESAMIENTO = CARPETA / "modelo.joblib"
RED = CARPETA / "red.keras"


def cargar_modelo(ruta_prepro: Path = PREPROCESAMIENTO, ruta_red: Path = RED) -> dict:
    """Carga todo lo aprendido en el entrenamiento.

    modelo.joblib es un diccionario con los imputadores (medianas, modas,
    KMeans), el escalador, las tablas fijas (ángulos del viento y
    coordenadas de las ciudades), las columnas de entrada de la red y su
    umbral. red.keras es la red neuronal entrenada.

    Returns:
        El diccionario de modelo.joblib con la red agregada en "red".
    """
    art = joblib.load(ruta_prepro)
    art["red"] = tf.keras.models.load_model(ruta_red)
    return art


def preprocesar(datos: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Aplica a datos crudos los mismos pasos que en el notebook.

    Solo usa transform y predict: todo lo aprendido viene en art, nunca
    se vuelve a hacer fit con los datos que se quieren predecir.
    """
    faltantes = [c for c in art["columnas_entrada"] if c not in datos.columns]
    if faltantes:
        raise ValueError(f"le faltan las columnas {', '.join(faltantes)}")

    X = datos[art["columnas_entrada"]].copy()

    # 1. Humedad en 0 y nubosidad en 9 son errores de carga: pasan a faltantes
    for col in ["Humidity9am", "Humidity3pm"]:
        X.loc[X[col] == 0, col] = np.nan
    for col in ["Cloud9am", "Cloud3pm"]:
        X.loc[X[col] == 9, col] = np.nan

    # 2 y 3. Mediana y moda de entrenamiento
    X[art["continuas_mediana"]] = X[art["continuas_mediana"]].fillna(art["medianas"])
    X[art["categoricas_moda"]] = X[art["categoricas_moda"]].fillna(art["modas"])

    # 4 y 5. Sunshine y nubosidad con el valor del cluster de cada día
    vars_clusters = list(art["scaler_clusters"].feature_names_in_)
    clusters = art["kmeans"].predict(art["scaler_clusters"].transform(X[vars_clusters]))
    clusters = pd.Series(clusters, index=X.index)
    for col in ["Sunshine", "Cloud9am", "Cloud3pm"]:
        X[col] = X[col].fillna(clusters.map(art["valores_por_cluster"][col]))

    # 6. Dirección del viento a seno y coseno
    angulos = art["angulos_viento"]
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        desconocidas = set(X[col]) - set(angulos)
        if desconocidas:
            raise ValueError(f"{col} tiene direcciones desconocidas: {', '.join(map(str, desconocidas))}")
        radianes = np.radians(X[col].map(angulos))
        X[col + "_seno"] = np.sin(radianes)
        X[col + "_coseno"] = np.cos(radianes)

    # 7. RainToday a 0 / 1
    X["RainToday"] = (X["RainToday"] == "Yes").astype(int)

    # 8. Location a latitud y longitud
    coords = art["coords_ciudades"]
    desconocidas = set(X["Location"]) - set(coords)
    if desconocidas:
        raise ValueError(f"hay ciudades desconocidas: {', '.join(map(str, desconocidas))}")
    X["Lat"] = X["Location"].map(lambda c: coords[c][0])
    X["Lon"] = X["Location"].map(lambda c: coords[c][1])

    # 9 y 10. El escalador se ajustó también con LogRainfall, así que hay que calcularla
    X["LogRainfall"] = np.log1p(X["Rainfall"])
    a_escalar = list(art["x_scaler"].feature_names_in_)
    X[a_escalar] = art["x_scaler"].transform(X[a_escalar])

    # 11. Las columnas y su orden salen del notebook (columnas_modelo), no de una lista escrita a mano
    return X[art["columnas_modelo"]]


def predecir(datos: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Devuelve la probabilidad de lluvia y la predicción con el umbral."""
    X = preprocesar(datos, art)
    prob = art["red"].predict(X.to_numpy(dtype="float32"), verbose=0).ravel()
    return pd.DataFrame({"prob_lluvia": prob.round(3),
                         "RainTomorrow_pred": np.where(prob >= art["umbral"], "Yes", "No")}, index=datos.index)


def main(entrada: str, salida: str) -> None:
    """Lee entrada, predice cada fila y guarda el resultado en salida."""
    art = cargar_modelo()
    datos = pd.read_csv(entrada)
    try:
        predicciones = predecir(datos, art)
    except ValueError as error:
        sys.exit(f"Error en {entrada}: {error}")
    datos[predicciones.columns] = predicciones
    datos.to_csv(salida, index=False)
    print(f"{len(datos)} filas predichas en {salida}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python inferencia.py <entrada.csv> <salida.csv>")
    main(sys.argv[1], sys.argv[2])
