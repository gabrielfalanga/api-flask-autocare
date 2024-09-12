import tensorflow_hub as hub
import tensorflow_text
from corpus import frases_pastilhas_freio
from corpus import frases_velas_ignicao
from corpus import frases_transmissao


# carregando modelo
try:
    model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")
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
    # criando lista de treino no formato:
    # [('classificação do problema', frase, embedding da frase)]

    embeddings_pastilhas = [('pastilhas_de_freio', frase, embed(frase)) for frase in frases_pastilhas_freio]

    embeddings_velas = [('velas_de_ignicao', frase, embed(frase)) for frase in frases_velas_ignicao]

    embeddings_transmissao = [('transmissao', frase, embed(frase)) for frase in frases_transmissao]

    embeddings_alinhamento = [("alinhamento", frase, embed(frase)) for frase in frases_alinhamento]

    embeddings_treino = embeddings_pastilhas + embeddings_velas + embeddings_transmissao + embeddings_alinhamento
    return embeddings_treino
