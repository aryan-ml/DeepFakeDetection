## Comments are done by me so chill :)

import os                           ## for file folder operation
import pandas as pd     
import cv2                          ## read write video frames
from mtcnn import MTCNN             ## depends on Tensorflow 
import glob                         ## searching file paths


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  ## connecting to root dir 


REAL_DIR = os.path.join(BASE_DIR, "data/raw/original_sequences/youtube/c23/videos")             ## Path of real set
FAKE_DIR = os.path.join(BASE_DIR, "data/raw/manipulated_sequences/Deepfakes/c23/videos")        ## Path of Fake set


OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/faces")             ## Save location
CSV_PATH = os.path.join(BASE_DIR, "data/processed/manifest.csv")        ## CSV location


FRAME_INTERVAL = 5                          ## Take every 5th frame

FACE_SIZE = 224                             ## Default (224) used for standard input for models 

detector = MTCNN()                          ## Initalizing MTCNN (Multi-task convolutional neural network)



os.makedirs(OUTPUT_DIR, exist_ok=True)   ## creates dirs if not already created 
os.makedirs(CSV_PATH, exist_ok=True)        

data_row = []                               ## List to store CSV rows




def extract_faces_from_video(video_path, label):
    video_id = os.path.splitext()