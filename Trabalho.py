import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (224, 224)
face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')

def CriaDiretorios():
    try:       
        # creating a folder named data 
        if not os.path.exists('Pessoas'):
            os.makedirs('Pessoas')
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of Pessoas') 
    try:       
        # creating a folder named data 
        if not os.path.exists('Hist'):
            os.makedirs('Hist')
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of Hist')
    try:       
        # creating a folder named data 
        if not os.path.exists('HistFace'):
            os.makedirs('HistFace')
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of HistFace')

def ComputaHist(img, currentframe, local):
    plt.hist(img.ravel(),256,[0,256])
    name = local +'/frame' + str(currentframe) + '.jpg'
    plt.savefig(name)

def ComputaFace(img, currentframe):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        name = './Pessoas/frame' + str(currentframe) + '.jpg'
        cv2.imwrite(name, img)
        #Salva histograma das fotos
        ComputaHist(img, currentframe, "HistFace")

CriaDiretorios()
video = cv2.VideoCapture("video/neo.mp4")
currentframe = 0

while(True):       
    ret,frame = video.read()       
    if ret:
        print(currentframe)
        #Salva faces
        ComputaFace(frame, currentframe)
        #SAlva todos os histogramas
        ComputaHist(frame, currentframe, "Hist")
        currentframe += 1
    else: 
        break     

video.release() 
cv2.destroyAllWindows()