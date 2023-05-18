import streamlit as st
import numpy as np
from PIL import Image     #THIS one helps to insert image in streamlit...
st.title("My new WEB page")
st.header("Welcome to my first web page using streamlit")
st.write("HOW are you all today?")
a = [1, 2, 3, 4, 5, 6]
n = np.array(a)
b = n.reshape((2, 3))
    # st.dataframe(b,width=500,height=500)
st.markdown("---")
c = {
        "Name": ["Vidhyadharan", "Siva", "Jaya", "Rakesh", "imitias", "Sunil"],
        "Age": [19, 18, 18, 17, 18, 19],
        "city": ["Chennai", "Kanchi", "Avadi", "Vellore", "Thiruvanamalai", "Chennai"]
    }
st.dataframe(c, width=550, height=500)
st.write(c)
st.json(c)
st.markdown("[my website](https://student.saveetha.ac.in)")  # Mardown("[tile](url of the website)")
st.markdown("---")
    # st.markdown("[AVERT](e6febe09-3122-41a9-8885-630b694e0c99.jpeg)")
    # st.image(e6febe09-3122-41a9-8885-630b694e0c99.jpeg, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # com.html("""
    # <img src="e6febe09-3122-41a9-8885-630b694e0c99".jpeg,width=400px,height=500px>

    # </img>
    # """)
img = Image.open('avert.jpeg')
st.image(img)  # image can inser in 2 ways..
    # 1.st.image()
    # 2.st.markdown()


