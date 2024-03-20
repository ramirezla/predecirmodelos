# Desarrollado por: Luis A, Ramirez G.<br>
# GitHub: ramirezla<br>
# Email: ramirezgluisalberto@gmail.com<br>
# Email: ramirezluisalberto@hotmail.com<br>

# Versiones paquetes utilizados: <br>
# OS: Linux x64 3.10.0-1160.92.1.el7.x8_64<br>
# Python 3.6.8<br>

import numpy as np
import pandas as pd

from fastapi import FastAPI

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def cargarDatos():
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
	
	return df_datos_compacto
	
# Cargando los set de datos desde el archivo .csv a un dataframe
df_datos_compacto = cargarDatos()

# Dividir los datos en conjunto de entrenamiento y prueba
train_text, test_text, train_labels, test_labels = train_test_split(df_datos_compacto["text"],
																	df_datos_compacto["label"],
																	test_size=0.33,
																	random_state=42,
																	stratify=df_datos_compacto["label"])	

### ---- INICIO Sin token ni vector ---- ###

# Crear el pipeline
pipeline = Pipeline([
	('vectorizer', TfidfVectorizer()),  # Vectorización de texto utilizando TF-IDF
	('classifier', LinearSVC())  		# Clasificador LinearSVC
])

# Entrenar el modelo sin tokenizador
pipeline.fit(train_text, train_labels)
# Evaluar el modelo sin tokenizador
y_pred = pipeline.predict(test_text)

### ---- FIN Sin token ni vector ---- ###

### ---- INICIO con token y vector ---- ###

# Tokenizar en palabras
def tokenize(sentence):
	import string
	
	punctuation = set(string.punctuation)
	tokens = []
	for token in sentence.split():
		new_token = []
		for character in token:
			if character not in punctuation:
				new_token.append(character.lower())
		if new_token:
			tokens.append("".join(new_token))
	return tokens
																	
# Tokenizando los datos.
real_vectorizer = CountVectorizer(tokenizer = tokenize, binary=True)
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)

# Código de inicialización, entrenamiento de modelos.
classifier = LinearSVC()
classifier.fit(train_X, train_labels)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
		  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
		  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
		  verbose=0)

### ---- FIN con token y vector ---- ###

app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')

@app.get('/predecir_modelo/{texto}')
def predecir_modelo(texto:str):
	try:
		cadena = [texto]
		predecir = pipeline.predict(cadena)
	except (ValueError, SyntaxError):
		pass
	return (list(predecir))
		  
@app.get('/predecir_modelo_tv/{texto}')
def predecir_modelo_tv(texto:str):
	try:
		texto_preprocesado = [texto]
		# Transformar el texto preprocesado utilizando el vectorizador cargado
		texto_transformado = real_vectorizer.transform(texto_preprocesado)
		# Realizar la predicción utilizando el modelo cargado
		prediccion = classifier.predict(texto_transformado)
	except (ValueError, SyntaxError):
		pass
	return (list(prediccion))  
