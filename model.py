import tensorflow_hub as hub
import tensorflow_text
import numpy as np


# carregando modelo
try:
    model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")
except Exception as e:
    print('Erro ao carregar o modelo: ', e)


# função que retorna os embeddings dos textos passados
def embed(texto):
    return model(texto)
