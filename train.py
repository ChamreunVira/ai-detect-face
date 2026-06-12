import os
import cv2
import numpy as np

# Initialize face detector and the LBPH face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = 'dataset'
face_samples = []
face_ids = []

# Mapping dictionary
label_map = {}
current_id = 0

print("Training started. Processing images...")

for root, dirs, files in os.walk(dataset_path):
    for dir_name in dirs:
        if dir_name not in label_map:
            label_map[dir_name] = current_id
            current_id += 1
            
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            numeric_id = label_map[person_name]
            
            # Load image and convert to grayscale
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray)
            
            # Detect face regions to ignore background noise
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                # Crop out the face
                face_samples.append(gray[y:y+h, x:x+w])
                face_ids.append(numeric_id)

# Train
recognizer.train(face_samples, np.array(face_ids))

# Save
recognizer.write('trainer.yml')

print(f"Training complete! Saved 'trainer.yml'.")
print("Mapping key for your prediction script:")
print(label_map)

