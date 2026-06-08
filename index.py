import cv2

// names = {0: "CheaUsa" , 1: "ChamreunVira" , 2: "Bunchhean"} 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml') # Load trained weights

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        # Predict
        id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # LBPH confidence score
        if confidence < 100:
            name = names.get(id_num, "Unknown")
            display_text = f"{name} ({round(100 - confidence)}%)"
        else:
            display_text = "Unknown"
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    cv2.imshow('Face Recognition Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
a
