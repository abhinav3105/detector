from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import colorama
app = Flask(__name__)

model=pickle.load(open('model1.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    Transaction_Amount = float(request.form['Transaction_Amount'])
    Average_Transaction_Amount = float(request.form['Average_Transaction_Amount'])
    Frequency_of_Transactions = float(request.form['Frequency_of_Transactions'])
    # spm = float(request.form['spm'])
    # input_data = [[so2,no2,rspm,spm]]
    input_data = [[Transaction_Amount,Average_Transaction_Amount,Frequency_of_Transactions]]
    user_anomaly_pred  = model.predict(input_data)[0]

    df = pd.DataFrame({'name': ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions'],
                   'age': [Transaction_Amount, Average_Transaction_Amount, Frequency_of_Transactions]})


    largest_value = df['age'].max()


    index_of_largest_value = df['age'].idxmax()


    largest_value_name = df.loc[index_of_largest_value, 'name']
    z = largest_value_name
    user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

    if user_anomaly_pred_binary == 1:
        return render_template('not.html')
    else:
        return render_template('verify.html')
if __name__ == '__main__':
    app.run()