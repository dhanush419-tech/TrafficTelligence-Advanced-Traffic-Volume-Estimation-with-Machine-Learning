from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result', methods=['POST'])
def result():
    try:
       
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        
        holiday_encoded = encoder.transform([holiday])[0]
        weather_encoded = encoder.transform([weather])[0]

      
        features = np.array([[holiday_encoded, temp, rain, snow, weather_encoded,
                              year, month, day, hours, minutes, seconds]])

        prediction = model.predict(features)

        output = round(prediction[0], 2)

        return render_template('result.html', 
                       result=f'Estimated Traffic Volume is: {output}')

    except Exception as e:
        return render_template('result.html',
                               prediction_text=f'Error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
