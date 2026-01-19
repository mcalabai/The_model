import cv2

#Read Image
image = cv2.imread('D:/USER/Antigravity/The_model/The_model/vnev/male_1.jpg')
image = cv2.resize(image, (720, 640))

#Define Model
face_pbtxt = "D:/USER/Antigravity/The_model/The_model/model/opencv_face_detector.pbtxt"
face_pb = "D:/USER/Antigravity/The_model/The_model/model/opencv_face_detector_uint8.pb"
age_prototxt = "D:/USER/Antigravity/The_model/The_model/model/age_deploy.prototxt"
age_model = "D:/USER/Antigravity/The_model/The_model/model/age_net.caffemodel"
gender_prototxt = "D:/USER/Antigravity/The_model/The_model/model/gender_deploy.prototxt"
gender_model = "D:/USER/Antigravity/The_model/The_model/model/gender_net.caffemodel"
MODEL_MEAN_VALUES= [104, 117, 123]

#Load Models 
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gen = cv2.dnn.readNet(gender_model, gender_prototxt)

#Setup Classifications
age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

#Copy Image
img_cp = image.copy()


#Get Iamge Dimensions & Blob
img_h = img_cp.shape[0]
img_w = img_cp.shape[1] 
blob = cv2.dnn.blobFromImage(img_cp,1.0, (300,300),MODEL_MEAN_VALUES, swapRB=True, crop=False)


face.setInput(blob)
detected_faces = face.forward()

face_bounds = []

#Draw Revtangle Over Faces
for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0,0,i,2]
    if confidence > 0.99:
        x1 = detected_faces[0,0,i,3] * img_w
        y1 = detected_faces[0,0,i,4] * img_h
        x2 = detected_faces[0,0,i,5] * img_w
        y2 = detected_faces[0,0,i,6] * img_h
        face_bounds.append([int(x1), int(y1), int(x2), int(y2)])
        cv2.rectangle(img_cp, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0),int(round(img_h/150)),8)
        face_bounds.append([x1, x2, y1, y2])    


for face_bound in face_bounds:
    try:
        face = img_cp[max(0, face_bound[1]-15): min(face_bound[3]+15,img_cp.shape[0]-1),
                    max(0, face_bound[0]-15): min(face_bound[2]+15,img_cp.shape[1]-1)
                     ]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=True)
        gen.setInput(blob)
        gender_prediction = gen.forward()
        gender = gender_classifications[gender_prediction[0].argmax()]


        age.setInput(blob)
        age_prediction = age.forward()
        age = age_classifications[age_prediction[0].argmax()]
        print(age)
        if (age == '(0-2)') or (age == '(4-6)') or (age == '(8-12)'):
            age_class = 'Not Eligible for social media'
        elif (age == '(15-20)') or (age == '(25-32)') or (age == '(38-43)') or (age == '(48-53)'):
            age_class = 'Eligible for social media'    
        print(age_class)

    except Exception as e :
        print(e)
        continue

cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()


