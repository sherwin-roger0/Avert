import requests
import streamlit as st
from PIL import Image
st.markdown("<h1 style='text-align: center';>Login/sign up</h1>",unsafe_allow_html=True)
with st.form('Login_form',clear_on_submit=True):
    image = Image.open("avert.jpeg")
    photo = st.image(image, width=100)
    cl1,cl2 =st.columns(2)
    fname=cl1.text_input("First Name")
    lname=cl2.text_input("Last Name")
    st.text_input("IMARDH NO")
    passw =st.text_input("Password",type='password')
    c_pass=st.text_input("Confirm Password")
    sub=st.form_submit_button("Submit")


if sub:
    if(fname=="" and lname==""):
        st.warning("Please fill the required field")
    elif passw!=c_pass:
        st.warning("Password not matching")
    else:
        st.success("Successfully submitted")
        requests.get("http://localhost:8502/")


