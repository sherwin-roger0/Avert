import os
import gridfs

from pymongo import MongoClient

URL = "mongodb+srv://MONGO:betterthanyou@cluster0.88weyxd.mongodb.net/?retryWrites=true&w=majority"



def mongo_conn():
    """create a connection"""
    try:
        conn = MongoClient(URL)
        print("Mongodb Connected", conn)
        return conn.Ytube
    except Exception as err:
        print(f"Error in mongodb connection: {err}")


def upload_file(file_loc, file_name, fs):
    """upload file to mongodb"""
    with open(file_loc, 'rb') as file_data:
        data = file_data.read()

    # put file into mongodb
    fs.put(data, filename=file_name)
    print("Upload Complete")

def download_file(download_loc, db, fs, file_name):
    """download file from mongodb"""
    data = db.student.files.find_one({"filename": file_name})

    fs_id = data['_id']
    out_data = fs.get(fs_id).read()

    with open(download_loc, 'wb') as output:
        output.write(out_data)

    print("Download Completed!")

db = mongo_conn()
fs = gridfs.GridFS(db, collection="student")

file_name = "converted.mp4"
file_loc = "" + file_name

def upload():
    upload_file(file_loc=file_loc, file_name=file_name, fs=fs)
