from ultralytics import YOLO
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

model = YOLO("yolov8n-pose.pt")

results = model("videos", save=True)


with open('keypoints.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['frame']
    for i in range(17):
        header += [f'x{i+1}', f'y{i+1}']
    writer.writerow(header)

    for frame_id, r in enumerate(results):
        if r.keypoints is not None:
            for person_kpts in r.keypoints: 
                kpts_flat = []
                for kp in person_kpts:  
                    kpts_flat += [kp[0], kp[1]]  
                writer.writerow([frame_id] + kpts_flat)

print("Keypoint CSV saved successfully!")

