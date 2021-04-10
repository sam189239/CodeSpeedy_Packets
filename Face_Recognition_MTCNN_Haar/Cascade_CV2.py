import cv2
import matplotlib.pyplot as plt

dir = 'H:\\Code\\current_files\\My_Code\\Face_Recognition'
img = cv2.imread(dir + '/123.jpg')

classifier = cv2.CascadeClassifier(dir + '/haarcascade_frontalface_alt.xml')
f_boxes = classifier.detectMultiScale(img,1.05,3)
for box in f_boxes:
   print(box)
   cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255, 0, 0), 1)

cv2.imshow('Detected!',img)
cv2.waitKey(0)
cv2.destroyAllWindows

