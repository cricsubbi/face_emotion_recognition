import cv2, time,os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

video = cv2.VideoCapture(0)

a = 1
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')

while True:
    a = a+1
    check, frame = video.read()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.03,minNeighbors = 5)
    for x,y,w,h in faces:
        gray_img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(46,211,79),thickness=2)
        pred_face = gray_img[y:y+w,x:x+h]
        pred_face=cv2.resize(pred_face,(48,48))
        img_pixels = image.img_to_array(pred_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(46,211,79),thickness=2)
    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)
    
    
    if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressedqqqqqqq
        break
    


video.release()

cv2.destroyAllWindows()