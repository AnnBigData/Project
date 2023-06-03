import pickle
from flask import Flask, jsonify
from flask import request
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

app = Flask(__name__)

# http://localhost:5000/
# GET
# POST
# PUT
# DELETE

# implementacja
class Perceptron():

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():
    # sepal length
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))

    features = [sepal_length,
                petal_length]

    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    X = df.iloc[:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 0, -1, 1)

    ppn = Perceptron(n_iter=20)
    ppn.fit(X, y)

    with open('model.pkl', "wb") as picklefile:
        pickle.dump(ppn, picklefile)

    # Load pickled model file
    with open('model.pkl', "rb") as picklefile:
        model = pickle.load(picklefile)

    # Predict the class using the model
    predicted_class = int(model.predict(features))

    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run()
