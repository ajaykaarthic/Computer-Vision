import numpy as np
import cv2 as cv

#define source video
cap =cv.VideoCapture(0)

#Cascade classifiers
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while(1):
    #capturing frame by frame
    ret,frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #detect face    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv.imshow('Output',frame)
    
    #out.write(frame)
    k =cv.waitKey(2)
    if k & 0xFF==27:
        break

        
    k=cv.waitKey(2)
    if k==27:
        break
cap.release()
cv.destroyAllWindows()
