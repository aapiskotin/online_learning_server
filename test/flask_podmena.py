from datetime import datetime, timedelta
from threading import Thread

from flask import Flask

app = Flask(__name__)

counter = 0

increment_time_interval = timedelta(seconds=5)
time_to_increment = datetime.now()

def increment_counter():
    global counter
    global time_to_increment
    if datetime.now() < time_to_increment:
        return
    counter += 1
    time_to_increment = datetime.now() + increment_time_interval

@app.route('/counter')
def get_counter():
    global counter
    thread = Thread(target=increment_counter)
    thread.start()
    return str(counter)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port='1337')
