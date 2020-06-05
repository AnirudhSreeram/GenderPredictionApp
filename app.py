import streamlit as st
import numpy as np
from sklearn.externals import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
gender_nv_model = open("/home/admin1/Documents/Gender_prediction/naivebayesgendermodel.pkl","rb")
count_vec = open("/home/admin1/Documents/Gender_prediction/Count_vector.pkl","rb")
clf = joblib.load(gender_nv_model)
cv = joblib.load(count_vec)

st.title("Gender Detection Using Name")

html_temp = """
<div>
<h2> My First application </h2> 
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
@st.cache
#def predict_gen(data):
#    vect = 


def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        result = "Female"
    else:
        result = "Male"
    return result    

name =st.text_input("Enter the name here","Type here")
if st.button("Classify"):
     st.text("Name {}".format(name.title())) 
     result1 = genderpredictor(name)

     st.success("Name {}, this name was classified as {}".format(name.title(),result1))
