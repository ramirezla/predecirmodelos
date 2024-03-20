# Desarrollado por: Luis A, Ramirez G.<br>
# GitHub: ramirezla<br>
# Email: ramirezgluisalberto@gmail.com<br>
# Email: ramirezluisalberto@hotmail.com<br>

# Versiones paquetes utilizados: <br>
# OS: Linux x64 3.10.0-1160.92.1.el7.x8_64<br>
# Python 3.6.8<br>

from fastapi import FastAPI

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def cargarDatos():
	import pandas as pd
	import numpy as np
	import csv
	
	# Cargando los datos desde un archivo
	df_datos = pd.read_csv("Marcas-Modelos_ICR_limpio.csv", encoding='latin-1', sep='|')[['snombremarca','smodelo','MODELO PADRE']]
	df_datos.dropna()
	# Modelando los datos
	df_datos_compacto = df_datos[['MODELO PADRE']]
	df_datos_compacto = df_datos_compacto.rename(columns={'MODELO PADRE':'label'})
	df_datos_compacto["text"] = None
	df_datos_compacto.loc[:,'text'] = df_datos['snombremarca'].astype(str) + ' ' + df_datos['smodelo'].astype(str)	
	# Creando un arreglo con cada lista de los datos.
	descripciones = np.array(df_datos_compacto["text"])
	modelos = np.array(df_datos_compacto["label"])
	
	return descripciones, modelos

descripcion, modelos = cargarDatos()

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(descripcion, modelos, test_size=0.2, random_state=42)

# Crear el pipeline
pipeline = Pipeline([
	('vectorizer', TfidfVectorizer()),  # Vectorización de texto utilizando TF-IDF
	('classifier', LinearSVC())  		# Clasificador LinearSVC
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = pipeline.predict(X_test)

app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')

@app.get('/predecir_modelo/{texto}')
def predecir_modelo(texto:str):
	try:
		cadena = [texto]
		predecir = pipeline.predict(cadena)
	except (ValueError, SyntaxError):
		pass
	return (list(predecir))
