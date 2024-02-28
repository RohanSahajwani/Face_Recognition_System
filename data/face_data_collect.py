import cv2
import numpy as np
cap = cv2.VideoCapture(0)

#Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"


while True:
    ret, frame = cap.read()
    #ret-> bool value, that checks whether frame is captured or not 
    
    if ret==False:
        continue
        
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detecting face
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    #1.3 -> scaling factor(as haarcascade training scale(fixed size) is different from detected face)
    #1.3 -> in each iteration, dimension is reduced by 30%
    #5 -> number of neighbours

    faces = sorted(faces, key = lambda f:f[2]*f[3])

    #pick the last face (because it has largest area)
    for face in faces[-1:]:
        #draw bounding box or the reactangle
        x,y,w,h = face
        cv2.rectangle(gray_frame, (x,y) , (x+w,y+h) , (0,255,255) , 2)

        #Extract (Crop out the required face) : Region of Interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_data))
    
    cv2.imshow("gray_frame", gray_frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    #key_pressed-> if person presses 'q', it can break . Wait key is in milli second

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

file_name = input("Enter the name of the person: ")
url = dataset_path+file_name+'.npy'

# Save this data into file system
np.save(url,face_data)
print("Data Successfully saved!! :")
    
cap.release()
cv2.destroyAllWindows()