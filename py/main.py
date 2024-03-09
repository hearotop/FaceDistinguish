import cv2
import numpy as np
import pymysql
from datetime import datetime
current_time = datetime.now() #记录获取人脸的时间
conn = pymysql.connect(host='192.168.10.30', user='hearo', password='ghb754869', database='face_recognition')
cursor = conn.cursor()
cap = cv2.VideoCapture(0)
face_id = 0
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\23757\AppData\Roaming\Python\Python312\site-packages\cv2\data\haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = gray[y:y + h, x:x + w]
        face_data = face_roi.tobytes()
        # 将人脸数据插入到数据库中
        insert_query = "INSERT INTO FaceData (FaceID , current_time,ImageData) VALUES (%s,%s,%s)"
        cursor.execute(insert_query, (face_id, current_time,face_data))
        conn.commit()
        face_id += 1 #实现图像的自增
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
