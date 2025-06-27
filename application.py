from flask import Flask,request,jsonify,render_template# for finding url of html
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and standaer sclaer pickle
ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_model=pickle.load(open('sclar.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
         
           
        Tem = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data=standard_model.transform([[Tem,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        re=ridge_model.predict(new_data)

        return render_template('home.html',results=re[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0") # your LAN (Wi-Fi or Ethernet IP), so other devices on the same Wi-Fi can access it
'''Ridge+Lassso+Elastic+Regression+Practicals
application'''