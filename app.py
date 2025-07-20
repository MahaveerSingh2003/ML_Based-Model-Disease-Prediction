import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

dt=pickle.load(open("decision.pkl","rb"))

def crop_recommend(arr):
    crop_name=dt.predict(arr)
    return crop_name[0]

st.title("Crop :seedling: recommendation on the basis of soil quality and season :rain_cloud:")

name=None
with st.form("my_form"):
    st.subheader("Provide these details of soil for prediction of crop")
    title_n = float(st.text_input('Ratio of Nitrogen content in soil', 90))
    title_p = float(st.text_input('Ratio of Phosphorus content in soil', 42))
    title_k = float(st.text_input('Ratio of Potassium content in soil', 43))
    title_temp = float(st.text_input('Temperature of that area in Celsius', 21))
    title_hum = float(st.text_input('Relative humidity in %', 82))
    title_ph = float(st.text_input('Ph value of the soil', 6.5))
    title_rain = float(st.text_input('Rainfall in that region in mm', 202))

    arr = np.array([[title_n, title_p, title_k, title_temp, title_hum, title_ph, title_rain]])

    submitted = st.form_submit_button("Recommend :rocket:")


if submitted:
    with st.spinner('Fetching details'):
        time.sleep(1.5)
    name=crop_recommend(arr)
    name=name.title()
    st.subheader("On the basis of above details the best suitable crop to cultivate according to our ML model is '{}' :seedling: ".format(name))