'''
Sistema de ejecución de comandos por medio de gestos y emociones a través de la captura de
imágenes del rostro utilizando WebCam, Python, Keras y OpenCV

'''
# Importando las librerias necesarias
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import webbrowser

# Cargando el archivo xml y h5
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

# Incializando las etiquetas que vamos a usar para las emociones
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# Iniciando la camara web con CV2
cap = cv2.VideoCapture(0)

# Inicializando los contadores de las emociones
langry = 0
lsad = 0
lneutral = 0
lsurprise = 0
lhappy = 0
linksAngry = ["https://www.youtube.com/watch?v=ZHPrM6VqASg","https://www.youtube.com/watch?v=DNrnDx-KZUY","https://www.youtube.com/watch?v=OnyjJDLcknA"]
linksSurprise = ["https://www.youtube.com/watch?v=QViMjIrO3Xo","https://www.youtube.com/watch?v=L6sQyjxxASs","https://www.youtube.com/watch?v=YALPoSuHsm0"]
linksSad = ["https://bemorewithless.com/sad/","https://www.youtube.com/watch?v=yyoOymMYHaU","https://www.youtube.com/watch?v=eqZwoLYEoHY"]
linksNeutral = ["https://www.youtube.com/watch?v=ZHPrM6VqASg","https://www.youtube.com/watch?v=QViMjIrO3Xo","https://www.youtube.com/watch?v=OnyjJDLcknA"]
# Contador de frames sacados
counter = 0

TIMER = 100
while True:
    # Toma solo un frame del video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    # Proceso de analisis y prediccion de la imagen
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # Hace una prediccion en el ROI

            preds = classifier.predict(roi)[0]
            #print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            #print("\nprediction max = ",preds.argmax())
            #print("\nlabel = ",label)
            if label == "Angry":
                langry += 1
            elif label == "Happy":
                lhappy += 1
            elif label == "Neutral":
                lneutral += 1
            elif label == "Sad":
                lsad += 1
            elif label == "Surprise":
                lsurprise += 1
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    counter += 1
    if counter >= TIMER:
        res = max(langry, lsad, lsurprise, lhappy, lneutral)

        if res == lneutral:
            webbrowser.open("https://www.youtube.com/watch?v=AUhOgfsDEmE")
        elif res == lsurprise:
            webbrowser.open("https://www.youtube.com/watch?v=ZA9kag0UllY")
        elif res == langry:
            webbrowser.open("https://www.youtube.com/watch?v=DNrnDx-KZUY")
        elif res == lsad:
            webbrowser.open("https://www.friv5online.com/es")
        counter = 0
        langry = 0
        lsad = 0
        lneutral = 0
        lsurprise = 0
        lhappy = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

total = langry + lsad + lsurprise + lhappy + lneutral

print("Estuviste " + str(round((langry*100)/total,2)) + " % del tiempo enojado.")
print("Estuviste " + str(round((lsad*100)/total,2)) + " % del tiempo triste.")
print("Estuviste " + str(round((lsurprise*100)/total,2)) + " % del tiempo sorprendido.")
print("Estuviste " + str(round((lhappy*100)/total,2)) + " % del tiempo feliz.")
print("Estuviste " + str(round((lneutral*100)/total,2)) + " % del tiempo sin alguna emocion reconocible.")
'''
if res == lneutral:
    webbrowser.open("https://www.youtube.com/watch?v=AUhOgfsDEmE")
elif res == lsurprise:
    webbrowser.open("https://www.youtube.com/watch?v=ZA9kag0UllY")
elif res == langry:
    webbrowser.open("https://www.youtube.com/watch?v=DNrnDx-KZUY")
elif res == lsad:
    webbrowser.open("https://www.friv5online.com/es")
'''
























