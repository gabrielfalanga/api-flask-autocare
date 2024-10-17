from joblib import load
import numpy as np
import pandas as pd
from os import environ


PATH_DO_MODELO = environ.get('PATH_RANDOM_FOREST_AUTOCARE')
RANDOM_FOREST_MODEL = load(PATH_DO_MODELO)

# carregando os arrays de categorias
marca_unique = np.load('arrays-variaveis-categorias/categorias_marca.npy', allow_pickle=True)
modelo_unique = np.load('arrays-variaveis-categorias/categorias_modelo.npy', allow_pickle=True)
combustivel_unique = np.load('arrays-variaveis-categorias/categorias_combustivel.npy', allow_pickle=True)
cambio_unique = np.load('arrays-variaveis-categorias/categorias_cambio.npy', allow_pickle=True)
cidade_unique = np.load('arrays-variaveis-categorias/categorias_cidade.npy', allow_pickle=True)
problema_unique = np.load('arrays-variaveis-categorias/categorias_problema.npy', allow_pickle=True)


def tratar_novos_dados(dados):
    # Convertendo os dados recebidos em DataFrame
    novos_dados = pd.DataFrame([dados])

    # Usando np.searchsorted para converter as categorias em códigos
    novos_dados['Marca'] = np.searchsorted(marca_unique, novos_dados['Marca'][0])
    novos_dados['Modelo'] = np.searchsorted(modelo_unique, novos_dados['Modelo'][0])
    novos_dados['Combustível'] = np.searchsorted(combustivel_unique, novos_dados['Combustível'][0])
    novos_dados['Câmbio'] = np.searchsorted(cambio_unique, novos_dados['Câmbio'][0])
    novos_dados['Cidade'] = np.searchsorted(cidade_unique, novos_dados['Cidade'][0])
    novos_dados['Problema'] = np.searchsorted(problema_unique, novos_dados['Problema'][0])
    return novos_dados


def calcular_orcamento(dados_veiculo):
    dados_tratados = tratar_novos_dados(dados_veiculo)
    orcamento = RANDOM_FOREST_MODEL.predict(dados_tratados)
    return float(orcamento)
