from datetime import datetime

from flask import Flask, jsonify, request

from app.model import Model

app = Flask(__name__)

model = Model()


@app.route('/test')
def hello_world():
    return 'This is a test response'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify(model.predict(data))


@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    model.update_model(data)
    return jsonify({'status': 'OK'})


@app.route('/time', methods=['GET', 'POST'])
def time():
    return str(datetime.now())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1337)
