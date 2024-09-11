from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# função que recebe entrada do usuário e classifica o problema
def classificar_entrada(entrada_usuario):
    return f"Resposta para: {entrada_usuario}"


# Rota para receber dados do frontend
@app.route('/classificar', methods=['POST'])
def classificar():
    data = request.json
    entrada = data.get('input')
    response = classificar_entrada(entrada_usuario=entrada)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
