from flask import Flask
from flask_cors import CORS     # this will avoid some problems with permissions

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World'

@app.route('/predict')
def predict():
    return 'predictions'


if __name__ == "__main__":
    app.run()