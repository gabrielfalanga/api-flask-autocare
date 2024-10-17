import tensorflow_hub as hub
import tensorflow_text
from corpus import frases_pastilhas_freio, frases_velas_ignicao, frases_transmissao, frases_alinhamento, \
frases_superaquecimento_motor, frases_bateria, frases_cambio
from os import environ


# carregando modelo
PATH_DO_MODELO = environ.get('PATH_LANGUAGE_MODEL_AUTOCARE')
try:
    # model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")
    model = hub.load(PATH_DO_MODELO)
except Exception as e:
    print('Erro ao carregar o modelo: ', e)


# função que retorna os embeddings dos textos passados
def embed(texto):
    return model(texto)


def criar_embeddings_treino():
    global frases_pastilhas_freio
    global frases_velas_ignicao
    global frases_transmissao
    global frases_alinhamento
    global frases_superaquecimento_motor
    global frases_bateria
    global frases_cambio
    # criando lista de treino no formato:
    # [('classificação do problema', frase, embedding da frase)]

    embeddings_pastilhas = [('pastilhas_de_freio', frase, embed(frase)) for frase in frases_pastilhas_freio]

    embeddings_velas = [('velas_de_ignicao', frase, embed(frase)) for frase in frases_velas_ignicao]

    embeddings_transmissao = [('transmissao', frase, embed(frase)) for frase in frases_transmissao]

    embeddings_alinhamento = [("alinhamento", frase, embed(frase)) for frase in frases_alinhamento]

    embeddings_superaquecimento = [("superaquecimento_motor", frase, embed(frase)) for frase in frases_superaquecimento_motor]

    embeddings_bateria = [("bateria", frase, embed(frase)) for frase in frases_bateria]

    embeddings_cambio = [("cambio", frase, embed(frase)) for frase in frases_cambio]

    embeddings_treino = (embeddings_pastilhas + embeddings_velas + embeddings_transmissao + embeddings_alinhamento + embeddings_superaquecimento
    + embeddings_bateria + embeddings_cambio)

    return embeddings_treino


if __name__ == '__main__':
    lista = criar_embeddings_treino()
    select = [item[0] for item in lista if item[0] == 'superaquecimento_motor']
    print(select)
