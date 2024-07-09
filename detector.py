import glob
import cv2
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import fiftyone as fo
from sklearn.model_selection import train_test_split

class ObjectDetection:
    def __init__(self, video_folder, output_folder):
        self.video_list = glob.glob(video_folder)
        self.output_folder = output_folder

    def video_to_frames(self, video_path):
        
        # split video file into frames and save them inside the folder
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        video_name = video_path.split("/")[-1].split(".")[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"{self.output_folder}/{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def video_processing(self):
        
        # processing all videos into frames and save them as image
        
        os.makedirs(self.output_folder, exist_ok=True)
        for video_path in self.video_list:
            self.video_to_frames(video_path)

    def load_model(self, model_type, model_name, model_classes=None):
        
        # load pretrained model for object detection
        
        model = torch.hub.load(model_type, model_name, pretrained=True)
        if model_classes:
            model.classes = model_classes
        return model

    def batch_list(self, input_list, batch_size):
        
        # split into batches
        
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    def model_step(self, model, batch_file_list):

        # get image objects coordinates using loaded pretrained model
        
        final_df = []

        for batch in batch_file_list:
            results = model(batch)

            t_list = []

            for i in range(len(batch)):
                if not results.pandas().xyxyn[i].empty:
                    for _ in range(len(results.pandas().xyxyn[i])):
                        t_list.append(batch[i])
                else:
                    os.remove(os.path.join(self.output_folder, batch[i]))

            df = pd.concat(results.pandas().xyxyn, ignore_index=True)
            df['filepath'] = t_list
            final_df.append(df)

        return final_df