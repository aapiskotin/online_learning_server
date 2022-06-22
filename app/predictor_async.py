import atexit
import dbm
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep

import ujson
from flask import Flask, request

from app.model import Model

app = Flask(__name__)

model = Model(load_model_path='last_execution.model')

cur_path = os.path.dirname(os.path.realpath(__file__))
data_path = Path(cur_path, '..', 'data')


def make_update(context, labels):
    return {
        'context': context,
        'label':   labels,
    }


def update_model(data: dict):
    sleep(1)
    print('Updating model with id {}'.format(data['id']))
    with dbm.open(str(data_path.joinpath('contexts')), 'c') as contexts_db:
        contexts_db[str(data['id'])] = ujson.dumps(data['context'])
        with dbm.open(str(data_path.joinpath('labels')), 'c') as labels_db:
            count = 0
            for id_ in contexts_db.keys():
                if cur_labels := labels_db.get(id_, None):
                    model.update_model(
                        make_update(
                            ujson.loads(contexts_db[id_]),
                            ujson.loads(cur_labels),
                        ),
                    )
                    del labels_db[id_]
                    del contexts_db[id_]
                    count += 1
                    print('Model updated with id {}'.format(id_))
                if count >= 2:
                    break


@app.route('/test')
def hello_world():
    return 'This is a test response'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    thread = Thread(target=update_model, args=(data,))
    thread.start()
    return model.predict(data['context'])


@app.route('/time', methods=['GET', 'POST'])
def time():
    return str(datetime.now())


def save_model():
    print('Saving model')
    model.model.save('1last_execution.model')

def finish_model():
    print('Finishing model')
    model.model.finish()


@app.route('/save', methods=['GET', 'POST'])
def save():
    save_model()
    return 'Model saved'


if __name__ == '__main__':
    atexit.register(finish_model)
    app.run(debug=True, host='0.0.0.0', port=228)
