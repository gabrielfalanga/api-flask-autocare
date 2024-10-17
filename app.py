from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import language_model
import random_forest_model
from oficinas import oficinas_cadastradas
import locale


# Configurando o locale para o padrão brasileiro
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

app = Flask(__name__)
CORS(app)
embeddings_treino = language_model.criar_embeddings_treino()


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
    elif tipo == 'alinhamento':
        return ('Pelo que você descreveu, o problema parece estar no alinhamento do veículo. Ele pode estar puxando '
                'para um dos lados, o que indica que o alinhamento das rodas está fora do ideal.')
    elif tipo == 'superaquecimento_motor':
        return ('Pelo que você descreveu, parece que o problema está relacionado ao superaquecimento do motor. '
            'Isso pode estar acontecendo devido à falta de fluido de arrefecimento, problemas no radiador ou '
            'no sistema de ventilação.')
    elif tipo == 'bateria':
        return ('Pelo que você disse, parece que o problema está relacionado à bateria do seu veículo. '
                'Isso pode ocorrer quando a bateria está no fim de sua vida útil ou quando há falhas no sistema elétrico.')
    elif tipo == "cambio":
            return ('Pelo que você descreveu, o problema parece estar no câmbio do veículo. Isso pode incluir dificuldades ao engatar as marchas, '
            'sensação de trancos, entre outras irregularidades. É recomendável fazer uma revisão completa do sistema de câmbio.')
    elif tipo == 'nao_identificado':
        return 'Desculpe, não entendi. Você poderia descrever sua situação mais detalhadamente?'


# função que recebe entrada do usuário e classifica o problema
def classificar_entrada(texto_entrada):
    global embeddings_treino

    if len(texto_entrada.split()) > 2:

        # computando embeddings da entrada do usuário
        emb_entrada = language_model.embed(texto_entrada)

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


def identificar_servico(tipo_problema, orcamento):
    if tipo_problema == 'pastilhas_de_freio':
        return ('Para resolver, pode ser feita uma revisão por um mecânico parceiro. A troca das pastilhas de freio '
                f'do seu carro, incluindo peças e mão de obra, custaria em torno de {locale.currency(orcamento, grouping=True)}.')
    elif tipo_problema == 'velas_de_ignicao':
        return ('Como resolução, pode ser agendada uma revisão em uma oficina parceira. A revisão e o ajuste '
                f'necessários custarão aproximadamente {locale.currency(orcamento, grouping=True)}.')
    elif tipo_problema == 'transmissao':
        return ('A solução é simples! Você pode agendar uma revisão com uma de nossas oficinas parceiras. '
                f'O serviço completo, incluindo a troca do fluido de transmissão custa em torno de {locale.currency(orcamento, grouping=True)}.')
    elif tipo_problema == 'alinhamento':
        return ('Para resolver esse problema é muito simples. Caso queira você pode realizar um agendamento em nossas oficinas parceiras. '
                f'O serviço completo custa em média {locale.currency(orcamento, grouping=True)}.')
    elif tipo_problema == 'superaquecimento_motor':
        return ('Para resolver, pode ser agendada uma revisão '
                f'em uma de nossas oficinas parceiras. O custo do conserto é em torno de {locale.currency(orcamento, grouping=True)}, dependendo '
                'da causa específica.')
    elif tipo_problema == 'bateria':
        return('Para resolver, você pode agendar uma troca em uma oficina parceira. '
               f'O custo para a resolução é de aproximadamente {locale.currency(orcamento, grouping=True)}.')
    elif tipo_problema == "cambio":
        return ('Para resolver, você pode agendar uma inspeção ou manutenção em uma de nossas oficinas parceiras. ' 
                f'O custo do conserto é aproximadamente {locale.currency(orcamento, grouping=True)}, dependendo da gravidade do problema e das peças necessárias.')
    elif tipo_problema == 'nao_identificado':
        return ''


# rota para receber dados do frontend
@app.route('/classificar', methods=['POST'])
def classificar():
    dados = request.json
    entrada = dados.get('input')
    tipo, response = classificar_entrada(texto_entrada=entrada)
    dados = {
        "Marca": "FORD",
        "Modelo": "FIESTA",
        "Ano": 2017,
        "Câmbio": "Manual",
        "Combustível": "Gasolina",
        "Quilometragem": 29764,
        "Cidade": "Rio de Janeiro",
        "Problema": tipo
    }
    if tipo != 'nao_identificado':
        orcamento = random_forest_model.calcular_orcamento(dados)
        print(orcamento)
    else:
        orcamento = 0.0
    return jsonify({'tipo': tipo, 'response': response, 'servico': identificar_servico(tipo, orcamento),
                    'orcamento': locale.format_string('%.2f', orcamento, grouping=True)})


def buscar_oficinas_disponiveis(data: str):
    global oficinas_cadastradas

    oficinas_disponiveis = []

    for oficina in oficinas_cadastradas:
        for data_dic in oficina['disponibilidade']:
            if data_dic['data'] == data:
                oficinas_disponiveis.append(oficina)

    return oficinas_disponiveis


@app.route('/buscar/oficinas', methods=['POST'])
def buscar_oficinas():
    dados = request.json
    data = dados.get('data')
    oficinas_disponiveis = buscar_oficinas_disponiveis(data)

    # cria a chave 'horarios' somente com os horários da data buscada
    for oficina in oficinas_disponiveis:
        for data_dic in oficina['disponibilidade']:
            if data_dic['data'] == data:
                oficina['horarios'] = data_dic['horarios']
                break

    return jsonify({'oficinas': oficinas_disponiveis})


if __name__ == '__main__':
    app.run(debug=True)
