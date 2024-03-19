# Desarrollado por: Luis A, Ramirez G.<br>
# GitHub: ramirezla<br>
# Email: ramirezgluisalberto@gmail.com<br>
# Email: ramirezluisalberto@hotmail.com<br>

# Versiones paquetes utilizados: <br>
# OS: Linux x64 3.10.0-1160.92.1.el7.x8_64<br>
# Python 3.6.8<br>

# fastapi==0.93.0<br>
# pandas==1.3.5<br>
# pip==23.2.1<br>
# scikit-learn==1.0.2<br>
# tokenizers==0.12.1<br>
# torch==1.10.2<br>
# transformers==4.18.0<br>

from fastapi import FastAPI
import pickle
#import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
# import sys


# Cargar el modelo y el vectorizador desde archivos
# with open('model_tokenizador_descripcion_a_modelos.pkl', 'rb') as f:
#    classifier = pickle.load(f)

with gzip.open('model_tokenizador_descripcion_a_modelos.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

with open('vectorizador_tokenizador_descripcion_a_modelos.pkl', 'rb') as f:
    real_vectorizer = pickle.load(f)

Tokenizar en palabras
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

# Se instancia una variable de tipo FastAPI
#app = FastAPI(title='ML_predecir_usar_modelo-main', description='Luis A Ramirez G')
app = FastAPI()
	
@app.get('/predecir_modelo/{texto}')
def predecir_modelo(texto: str):
    try:
		# Preprocesar el texto de ejemplo utilizando el mismo tokenizador
		texto_preprocesado="chevrolet ave 4 ptas"
		# Transformar el texto preprocesado utilizando el vectorizador cargado
		##texto_transformado=real_vectorizer.transform(texto_preprocesado)
		# Realizar la predicción utilizando el modelo cargado
		##prediccion=classifier.predict(texto_transformado)
    except (ValueError, SyntaxError):
        pass 
	return texto_preprocesado
