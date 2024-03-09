'''
本文件用来比较是否是同一个人的信息
'''
import cv2
import numpy as np
import pymysql
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = 0  # 初始化人脸ID
known_face_ids = set()  # 用于存储已知的人脸ID
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = gray[y:y + h, x:x + w]

        # 提取人脸特征
        label, confidence = face_recognizer.predict(face_roi)

        if label is not None:
            # 如果是已知人脸，则使用相应的face_id
            face_id = label
            print("Known face with ID:", face_id)
        else:
            # 如果是新的人脸，则分配新的face_id
            face_id = max(known_face_ids) + 1 if known_face_ids else 0
            print("New face detected, assigning ID:", face_id)
            # 保存新的人脸特征到数据库（此处省略代码）
            # ...
            known_face_ids.add(face_id)
        # 在此处将人脸数据和face_id插入到MySQL数据库（省略具体代码）
        # ...

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


