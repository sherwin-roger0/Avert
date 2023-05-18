import pickle
from pathlib import Path

import streamlit_authenticator as stu

names = ["Vidhydharan","Naveen","Siva","Rakesh","Varsha"]
username = {1:"vdharan",2:"naveen",3:"sive",4:"rakesh",5:"Vshernon"}
passwords = ["abc123","def456","ghi123","jkl123","vss123"]

hashed_pass = stu.Hasher(passwords).generate()

file_path = Path(__file__).parent /"hashed_pw.pkl"
with file_path.open("wb")as file:
    pickle.dump(hashed_pass,file)

