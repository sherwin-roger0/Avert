import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stu
from PIL import Image

names = ["Vidhyadharan","Naveen","Siva","Rakesh"]
username = ["vdharan","naveen","siva","rakesh"]
file_path=Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb")as file:
    hashed_pass = pickle.load(file)

authenticator = stu.Authenticate(names=names,usernames=username , passwords=hashed_pass, cookie_name="avert_dashboard",key="abcdef")
name,authentication_status,usernames = authenticator.login("Login","main")

if authentication_status == False:
    st.error("Username or password is incorrect")
if authentication_status == None:
    st.warning("Please Enter username and password")
if authentication_status == True:
    print("true")

    start = st.container()
    img = Image.open('avert.jpeg')
    st.image(img)
    #report=st.container()
    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f"Welcome {name}")


    user = st.container()
    widgest = st.container()
    neg = st.sidebar.radio("Crime Category", options=("Low Level", "High Level"))
    with start:
        st.title("CRIME Analytic")

    with user:
        st.header("What you think?")
        crime_tell = st.columns(1)
        crime_inp = st.slider('what will be the max-level', max_value=500, min_value=10, value=50, step=50)

        crime_chk = st.selectbox('How much can you Reduce?', options=[100, 200, 300, 'NO-LIMIT'], index=0)

        crime = st.text_input("What you think about Avert", max_chars=500)

        crime_opi = st.checkbox('I AGREE to the Terms and conditons')
        hell = st.button('OK')

    with widgest:
        file = st.file_uploader("Upload your file here:", type=['png', 'jpeg', 'jpg'])
        st.write(file)
