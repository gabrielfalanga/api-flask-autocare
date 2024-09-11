from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from model import embed
from model import criar_embeddings_treino

app = Flask(__name__)
CORS(app)
embeddings_treino = criar_embeddings_treino()


def criar_mensagem_problema(tipo):
    if tipo == 'pastilhas de freio':
        return 'Aah, claro! Ao que tudo indica, o problema do seu veículo é o desgaste das pastilhas de freio.'
    elif tipo == 'velas de ignição':
        return 'Pelo que você disse, parece que o problema do seu veículo está nas velas de ignição.'
    elif tipo == 'transmissão':
        return 'De acordo com meu conhecimento, a causa do que está acontecendo é um problema na transmissão do veículo.'
    elif tipo == 'não identificado':
        return 'Desculpe, não entendi. Você poderia descrever sua situação mais detalhadamente?'


# função que recebe entrada do usuário e classifica o problema
def classificar_entrada(texto_entrada):
    global embeddings_treino

    # computando embeddings da entrada do usuário
    emb_entrada = embed(texto_entrada)

    # criando lista com a similaridade da entrada com cada uma das frases de treino e suas classificações
    resultado = []
    for i in embeddings_treino:
        similaridade = np.inner(emb_entrada, i[2])
        resultado.append([i[0], similaridade])

    # ordenando a lista resultado a partir da maior similaridade
    classificacao = sorted(resultado, key=lambda x: x[1], reverse=True)

    tipo = classificacao[0][0] if classificacao[0][1] > 0.35 else 'não identificado'
    resposta = criar_mensagem_problema(tipo)

    return resposta


# Rota para receber dados do frontend
@app.route('/classificar', methods=['POST'])
def classificar():
    data = request.json
    entrada = data.get('input')
    response = classificar_entrada(texto_entrada=entrada)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
