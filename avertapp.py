import streamlit as st
from PIL import Image


def main():
    st.markdown("<h1 'style:text-align=center'>WELCOME TO AVERT</h1>",unsafe_allow_html=True)
    img = Image.open("avert.jpeg")
    st.image(img,width=100)
    st.markdown("<h2 'style:text-align=center'>Loading...</h2>",unsafe_allow_html=True)

main()