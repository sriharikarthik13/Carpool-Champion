"""Flask application to run the Carpool Champion Project"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carpool_predictor import carpool_prediction,carpool_time_effect

from flask import Flask, render_template, request




app = Flask(__name__) 
@app.route('/')
def index():
    """View for the form to submit for viewing the effect of carpooling and traffic prediction""" 
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    """View for the results of the project"""
    user_input = request.form['user_input']
    option = request.form['option']
    if option == "Option 1":
        value1,value2,value3 = carpool_prediction(int(user_input))
        value2 = round(value2)
        value3 = round(value3)
        return render_template('result1.html', value1=value1,value2=value2,value3=value3) 
    else:
        value1,value2,value3,value4,value5,value6 = carpool_time_effect(int(user_input))
        value3 = round(value3)
        value4 = round(value4)
        value5 = round(value5)
        value6 = round(value6)
        return render_template('result.html', value2=value2, value1=value1, value3=value3, value4=value4, value5=value5,value6=value6)

if __name__ == '__main__':
    app.run(debug=True)
