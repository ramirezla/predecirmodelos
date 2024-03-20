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
app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')

# Tokenizar en palabras
async def tokenize(sentence):
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

with open('model_tokenizador_descripcion_a_modelos.pkl', 'rb') as f:
 classifier = pickle.load(f)

with open('vectorizador_tokenizador_descripcion_a_modelos.pkl', 'rb') as f:
 real_vectorizer = pickle.load(f)

# app = FastAPI()
	
@app.get('/predecir_modelo/{texto}')
def predecir_modelo(texto: str):
 try:
  # Preprocesar el texto de ejemplo utilizando el mismo tokenizador
  texto_preprocesado = [texto]
  # Transformar el texto preprocesado utilizando el vectorizador cargado
  texto_transformado = real_vectorizer.transform(texto_preprocesado)
  # Realizar la predicción utilizando el modelo cargado
  prediccion = classifier.predict(texto_transformado)
 except (ValueError, SyntaxError):
  pass
 return {'Es: ':prediccion}
