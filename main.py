# Desarrollado por: Luis A, Ramirez G.<br>
# GitHub: ramirezla<br>
# Email: ramirezgluisalberto@gmail.com<br>
# Email: ramirezluisalberto@hotmail.com<br>

# Versiones paquetes utilizados: <br>
# OS: Linux x64 3.10.0-1160.92.1.el7.x8_64<br>
# Python 3.6.8<br>

from fastapi import FastAPI

import pickle
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys	

# Se instancia una variable de tipo FastAPI
#app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')

def cargarModelo():
	import pickle
	
	filename = "model_descripcion_a_modelos.pkl"
	pipeline = pickle.load(open(filename, "rb"))
	return pipeline

# Tokenizar en palabras
# filename = "model_descripcion_a_modelos.pkl"
# pipeline = pickle.load(open(filename, "rb"))

app = FastAPI()

@app.get("/")
def index():
    return {"Saludos": "Hola"}

@app.get("/predecir/{ejemplo}")
def predecir(ejemplo:str):
	return ejemplo

@app.get('/predecir_modelo')
def predecir_modelo():
	try:
		texto = "chevrolet ave0 lt"
		cadena = [texto]
		pipeline = cargarModelo()
		prediccion = texto
		predecir = pipeline.predict(cadena)
	except (ValueError, SyntaxError):
		pass
	return (prediccion + " " + predecir)
