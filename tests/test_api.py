from app import flask_app
# python -m pytest

client = flask_app.test_client()

def test_test():
    response = client.get("/test")
    assert response.json['result'] == 'ok'

def test_predict():

    response = client.post("/predict",
                           data={
                               'Sepal_Length': '0',
                               'Sepal_Width': '0',
                               'Petal_Length': '0',
                               'Petal_Width': '0'
                           })

    assert response.status_code == 200

    assert response.json['results'] == ['Versicolor']