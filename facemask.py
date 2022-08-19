
import numpy as np
import keras
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2

#Train CNN model
"""
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit_generator(
        training_set,
        epochs=8,
        validation_data=test_set,)
model.save('CNN_model.h5',model_saved)
"""
#Live Detection
mymodel=load_model('CNN_model.h5')

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _,faceimg=cap.read()
    face=face_cascade.detectMultiScale(faceimg,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face:
        face_img = faceimg[y:y+h, x:x+w]
        cv2.imwrite('current.jpg',face_img)
        test_image=image.load_img('current.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        pred=mymodel.predict(test_image)[0][0]
        if pred==1:
            cv2.putText(faceimg,'NO MASK Found',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            center = (x + w//2, y + h//2)
            cv2.ellipse(faceimg, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)


        else:
            cv2.putText(faceimg,'MASK Found',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            center = (x + w//2, y + h//2)
            cv2.ellipse(faceimg, center, (w//2, h//2), 0, 0, 360, (0, 255, 0), 3)

          
    cv2.imshow('img',faceimg)
    
    if cv2.waitKey(1)==ord('a'):
        break
    
cap.release()
cv2.destroyAllWindows()
