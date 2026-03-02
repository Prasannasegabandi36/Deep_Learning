import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import pandas as pd
import numpy as np


#load the trained model
model=tf.keras.models.load_model('model.h5')

#load the trained model, scaler pickle,onehot
model=load_model("model.h5")
#load the encoder and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)