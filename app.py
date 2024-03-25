import pickle

from flask import Flask,render_template,request,url_for
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model2.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    res = ''
    if prediction == 1:
        res = 'The person is suffering from heart diesease'
    else:
        res = "The person is not suffering from heart diesease"
    return render_template('index.html',prediction_text = res)

if __name__ == "__main__":
    app.run(debug=True)