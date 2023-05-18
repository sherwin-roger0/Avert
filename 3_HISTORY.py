import streamlit as st
from PIL import Image
import pandas as pd
start=st.container()
img=Image.open('avert.jpeg')
st.image(img)
##report=st.container()

user=st.container()
widgest=st.container()
neg = st.sidebar.radio("Crime Category", options=("Low Level", "High Level"))
with start:
    st.title("CRIME Analytic")

with user:
    st.header("What you think?")
    crime_tell=st.columns(1)
    crime_inp=st.slider( 'what will be the max-level',max_value=500,min_value=10,value=50, step=50)

    crime_chk = st.selectbox('How much can you Reduce?', options=[100, 200, 300, 'NO-LIMIT'], index=0)

    crime = st.text_input("What you think about Avert",max_chars=500)

    crime_opi=st.checkbox('I AGREE to the Terms and conditons')
    hell = st.button('OK')


with widgest:
    file=st.file_uploader("Upload your file here:",type=['png','jpeg','jpg'])
    st.write(file)
