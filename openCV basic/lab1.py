# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 18:14:11 2022
@author: dsouzm3
"""

import cv2 as cv

"""capturing and saving file"""
capture = cv.VideoCapture(0) #0 means capturing continuosly
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('alwyn.avi', fourcc, 20, (640,  480))

if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("end of video")
        break
    out.write(frame)
    cv.imshow("frame",frame)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
out.release()
cv.destroyAllWindows()

"""Reading from file"""
count = 0
cap = cv.VideoCapture('alwyn.avi')
while True:
    print(count)
    ret, frame = cap.read()
    if not ret:
        print("end of video: ", count)
        break
    count += 1
    if count==10:
        break
    if cv.waitKey(1) == ord('q'):
        break   

out = cv.imwrite('frame10.jpg', frame)
# cv.imshow("image", frame)
print("*"*20)
print(count)
frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("These are the frames: ", frames)
cap.release()
# out.release()

img = cv.imread('frame10.jpg')
cv.imshow("Cute Alwyn", img)
cv.waitKey(0)
cv.destroyAllWindows()
