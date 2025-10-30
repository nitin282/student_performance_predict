from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("math_score_model.joblib")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    reading_score = float(request.form['reading_score'])
    writing_score = float(request.form['writing_score'])
    Gender=int(request.form['Gender'])
    lunch_encode=int(request.form['lunch_encode'])
    test_encode=int(request.form['test_encode'])
    race_encode=int(request.form ['race_encode'])
    predicted_score = model.predict([[reading_score, writing_score,Gender,lunch_encode,test_encode,race_encode]])[0]
    return render_template('predict.html',prediction_text=f"Predicted Math Score: {round(predicted_score, 2)}")
if __name__ == '__main__':
    app.run(debug=True)
