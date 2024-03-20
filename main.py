# Desarrollado por: Luis A, Ramirez G.<br>
# GitHub: ramirezla<br>
# Email: ramirezgluisalberto@gmail.com<br>
# Email: ramirezluisalberto@hotmail.com<br>

# Versiones paquetes utilizados: <br>
# OS: Linux x64 3.10.0-1160.92.1.el7.x8_64<br>
# Python 3.6.8<br>

from fastapi import FastAPI

import pickle
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import string

# Se instancia una variable de tipo FastAPI
#app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')

# Tokenizar en palabras
filename = "model_descripcion_a_modelos.pkl"
pipeline = pickle.load(open(filename, "rb"))

app = FastAPI()
@app.get('/predecir_modelo/{texto}')
def predecir_modelo(texto: str):
 try:
  cadenaMarcaModelo = texto
  prediccion = pipeline.predict([cadenaMarcaModelo])
 except (ValueError, SyntaxError):
  pass
 return {'Es: ':prediccion}
