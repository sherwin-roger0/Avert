#Importing modulues

import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stu
from PIL import Image
import os
import cv2
import pandas as pd
from tensorflow import keras
from pymongo import MongoClient
import datetime

st.set_page_config(page_title="avert",page_icon="e6febe09-3122-41a9-8885-630b694e0c99.jpeg",layout="wide")

#Declaring names,username by default

names = ["Vidhyadharan","Naveen","Siva","Rakesh","Varsha"]
username = ["vdharan","naveen","siva","rakesh","Vshernon"]
file_path=Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb")as file:
    hashed_pass = pickle.load(file)
    
authenticator = stu.Authenticate(names=names,usernames=username,passwords=hashed_pass,cookie_name="avert_dashboard",key="abcdef")
name,authentication_status,usernames = authenticator.login("Login","main")

#Checking whether necessary information filled!

if authentication_status == False:
    st.error("Username or password is incorrect")
if authentication_status == None:
    st.warning("Please Enter username and password")
if authentication_status == True:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    MAX_SEQ_LENGTH = 100
    NUM_FEATURES = 2048


    st.markdown("<h1 style='text-align:center'>Avert</h1>", unsafe_allow_html=True)
    st.markdown("---")
    report=st.container()
    state=st.container()
    with report:
        st.header("Report for crime Case around the INDIA")

# Create a file uploader widget
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "webm"])

# Check if a file was uploaded
        if uploaded_file is not None:
    # Use the uploaded file in your app
            video_bytes = uploaded_file.read()
            st.video(video_bytes)
            with open("uploaded_video.mp4", "wb") as f:
                f.write(video_bytes)
        sequence_model = keras.models.load_model('saved_model')        
        def build_feature_extractor():
            feature_extractor = keras.applications.InceptionV3(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
            preprocess_input = keras.applications.inception_v3.preprocess_input

            inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
            preprocessed = preprocess_input(inputs)

            outputs = feature_extractor(preprocessed)
            return keras.Model(inputs, outputs, name="feature_extractor")


        feature_extractor = build_feature_extractor()

        # split the data into train and test

        import numpy as np
        df = pd.read_csv('dataset.csv', header=None)
        df.columns = ["class", "path"]
        df = df.astype({"class": str})
        train, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])
    
        train_df = train
        test_df = test

        print(f"Total videos for training: {len(train_df)}")
        print(f"Total videos for testing: {len(test_df)}")


        def crop_center_square(frame):
            y, x = frame.shape[0:2]
            min_dim = min(y, x)
            start_x = (x // 2) - (min_dim // 2)
            start_y = (y // 2) - (min_dim // 2)
            return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

        def load_video(path, max_frames=0, resize=(224, 224)):
            cap = cv2.VideoCapture(path)
            frames = []
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = crop_center_square(frame)
                    frame = cv2.resize(frame, resize)
                    frame = frame[:, :, [2, 1, 0]]
                    frames.append(frame)

                    if len(frames) == max_frames:
                        break
            finally:
                cap.release()
            return np.array(frames)
    
        label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df["class"])
        )
        print(label_processor.get_vocabulary())

        import imageio
        from tensorflow_docs.vis import embed

        def prepare_single_video(frames):
            frames = frames[None, ...]
            frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
            frame_featutes = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

            for i, batch in enumerate(frames):
                video_length = batch.shape[1]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])
                    frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

            return frame_featutes, frame_mask


        def sequence_prediction(path):
            class_vocab = label_processor.get_vocabulary()

            frames = load_video(os.path.join(path))
    
            frame_features, frame_mask = prepare_single_video(frames)
            probabilities = sequence_model.predict([frame_features, frame_mask])[0]
            v = ['Violence','NonViolence']
            metadata_class=[]
            metadata_probability=[]
            for i in np.argsort(probabilities)[::-1]:
                st.write(f"  {str(v[class_vocab[i].astype(int)])} :{class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
                metadata_class.append(class_vocab[i]) 
                metadata_probability.append(probabilities[i]*100) 

            cluster=MongoClient("mongodb+srv://MONGO:betterthanyou@cluster0.88weyxd.mongodb.net/?retryWrites=true&w=majority")
            db=cluster["test"]
            collection=db["student"]
            collection.insert_one({"class":metadata_class,"probability":metadata_probability,"date and time":datetime.datetime.now()})

            return frames
        if uploaded_file is not None:   
            test_frames = sequence_prediction("uploaded_video.mp4")
        










    st.text('This is the Date retrieve by Kaggle...'
    'This depicts the CRIME RATE in INDIA...This has helps us to analyis the cime data.')
st.markdown("---")

with state:
    st.markdown("<h1 'style=text-align: center'>TamilNadu city wise crime Data</h1>",unsafe_allow_html=True)
   
