# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup, redirect, url_for
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from joblib import load

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
# =========================================================================================

# Loading the fertility detection model
fertility_model_path = 'models/rf.joblib'
fertility_model = load(fertility_model_path)


# Custom functions for calculations
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def fertility_predict(data):
    """
    Predicts the fertility of the soil
    :params: data
    :return: prediction
    """
    prediction = fertility_model.predict(data)
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
# render home page

@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


@app.route('/fertility')
def fertility():
    title = 'Harvestify - Fertility Detection'
    return render_template('fertility.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)
    else:
        return redirect(url_for('crop_recommend'))

# render fertilizer recommendation result page
@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page

@app.route('/fertility-predict', methods=['POST'])
def fertility_prediction():
    # pH, EC, OC, OM, N, P, K, Zn, Fe, Cu, Mn, Sand, Silt, Clay, CaCO3, CEC handle post request
    title = 'Harvestify - Fertility Detection'
    # print(request.form.keys())
    ph = float(request.form['pH'])
    ec = float(request.form['EC'])
    oc = float(request.form['OC'])
    om = float(request.form['OM'])
    n = float(request.form['N'])
    p = float(request.form['P'])
    k = float(request.form['K'])
    zn = float(request.form['Zn'])
    fe = float(request.form['Fe'])
    cu = float(request.form['Cu'])
    mn = float(request.form['Mn'])
    sand = float(request.form['Sand'])
    silt = float(request.form['Silt'])
    clay = float(request.form['Clay'])
    caco3 = float(request.form['CaCO3'])
    cec = float(request.form['CEC'])
    # convert to numpy array
    data = np.array([[ph, ec, oc, om, n, p, k, zn, fe, cu, mn, sand, silt, clay, caco3, cec]])
    # predict
    prediction = fertility_predict(data)
    # convert to string
    if prediction == 0:
        prediction = "The soil sample is <u>fertile</u> and can be used for agriculture"
    else:
        prediction = "The soil sample is <u>infertile</u> and cannot be used for agriculture"
    return render_template('fertility-result.html', prediction=prediction, title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
