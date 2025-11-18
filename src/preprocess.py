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

data_rows = []                               ## List to store CSV rows




def extract_faces_from_video(video_path, label):
    video_id = os.path.splitext(os.path.basename(video_path))[0]        ## Just takes the name of the video file and no extention
    video_output_dir = os.path.join(OUTPUT_DIR, video_id)               ## Path 
    os.makedirs(video_output_dir, exist_ok=True)


    cap = cv2.VideoCapture(video_path)                                  ## Open the Video
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if frame_count % FRAME_INTERVAL == 0:                           ## Only take faces from every 5th frame
            rgb = cv2.cvtColor(frame, cv2.color_BGR2RGB)                ## OpenCV uses BGR and MTCNN uses RGB hence conversion is important


            faces = detector.detect_faces(rgb)                          ## Face detection model
            if len(faces) > 0:
                x, y, w, h = faces[0]["box"]                            ## Coordinates for detected face


                x, y = max(0, x), max(0, y)

                face_crop = rgb[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
                out_path = os.path.join(video_output_dir, f"{saved_count}.jpg")

                cv2.imwrite(out_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

                data_rows.append([out_path, label, video_id])           ## Sample gets added to a CSV
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {video_id}: {saved_count} faces saved.")



def process_folder(folder_path, label):
    videos = glob.glob(os.path.join(folder_path, "*.mp4"))
    for v in videos:
        extract_faces_from_video(v, label)



if __name__ == "__main__":
    print("Processing REAL videos...")
    process_folder(REAL_DIR, label=0)

    print("\nProcessing FAKE videos...")
    process_folder(FAKE_DIR, label=1)

    df = pd.DataFrame(data_rows, columns=["image_path", "label", "video_id"])
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved manifest CSV at {CSV_PATH}")