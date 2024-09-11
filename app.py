from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from model import embed
from model import criar_embeddings_treino

app = Flask(__name__)
CORS(app)
embeddings_treino = criar_embeddings_treino()


def criar_mensagem_problema(tipo):
    if tipo == 'pastilhas_de_freio':
        return ('Aah, claro! Ao que tudo indica, o problema do seu veículo é o desgaste das pastilhas de freio. '
                'Com o tempo, o desgaste dessas peças é comum.')
    elif tipo == 'velas_de_ignicao':
        return ('Pelo que você disse, parece que o problema do seu veículo está nas velas de ignição. Essa situação '
                'geralmente acontece por uma folga inadequada entre os eletrodos das velas.')
    elif tipo == 'transmissao':
        return ('De acordo com meu conhecimento, a causa do que está acontecendo é um problema na transmissão do '
                'veículo. Pode ser baixo nível de fluido de transmissão ou o fluido está sujo/contaminado.')
    elif tipo == 'nao_identificado':
        return 'Desculpe, não entendi. Você poderia descrever sua situação mais detalhadamente?'


# função que recebe entrada do usuário e classifica o problema
def classificar_entrada(texto_entrada):
    global embeddings_treino

    if len(texto_entrada.split()) > 2:

        # computando embeddings da entrada do usuário
        emb_entrada = embed(texto_entrada)

        # criando lista com a similaridade da entrada com cada uma das frases de treino e suas classificações
        resultado = []
        for i in embeddings_treino:
            similaridade = np.inner(emb_entrada, i[2])
            resultado.append([i[0], similaridade])

        # ordenando a lista resultado a partir da maior similaridade
        classificacao = sorted(resultado, key=lambda x: x[1], reverse=True)

        tipo = classificacao[0][0] if classificacao[0][1] > 0.35 else 'nao_identificado'
        resposta = criar_mensagem_problema(tipo)

    else:
        tipo = 'nao_identificado'
        resposta = criar_mensagem_problema(tipo)

    return tipo, resposta


def identificar_servico(tipo_problema):
    if tipo_problema == 'pastilhas_de_freio':
        return ('Para resolver, pode ser feita uma revisão por um mecânico parceiro. A troca das pastilhas de freio '
                'do seu carro, incluindo peças e mão de obra, custaria em torno de R$120,00.')
    elif tipo_problema == 'velas_de_ignicao':
        return ('Como resolução, pode ser agendada uma revisão em uma oficina parceira. A revisão e o ajuste '
                'necessários custarão aproximadamente R$65,00.')
    elif tipo_problema == 'transmissao':
        return ('A solução é simples! Você pode agendar uma revisão com uma de nossas oficinas parceiras. '
                'O serviço completo, incluindo a troca do fluido de transmissão custa em torno de R$400,00.')
    elif tipo_problema == 'nao_identificado':
        return ''


# Rota para receber dados do frontend
@app.route('/classificar', methods=['POST'])
def classificar():
    data = request.json
    entrada = data.get('input')
    tipo, response = classificar_entrada(texto_entrada=entrada)
    return jsonify({'tipo': tipo, 'response': response, 'servico': identificar_servico(tipo)})


@app.route('/buscar-oficinas', methods=['GET'])
def buscar_oficinas():
    lista_oficinas = [
        "Oficina do Betinho",
        "Oficina Porto Seguro",
        "André Autopeças"
    ]
    return jsonify({'oficinas': lista_oficinas})


@app.route('/buscar-datas', methods=['GET'])
def buscar_datas():
    lista_datas = [
        "12/09/24",
        "13/09/24",
        "16/09/24",
        "17/09/24",
        "18/09/24",
        "20/09/24",
        "21/09/24"
    ]
    return jsonify({'datas': lista_datas})


if __name__ == '__main__':
    app.run(debug=True)
