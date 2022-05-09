import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# PATH = r"4. Serve ML model over web API"

with open(f'model.pickle', 'rb') as handle:
    model = pickle.load(handle)

# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features).tolist()[0]

    #return {"results": prediction}
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction), input_values= str(features))

@flask_app.route("/test")
def test_ok():
    return {"result": "ok"}

if __name__ == "__main__":
    flask_app.run(debug=True)