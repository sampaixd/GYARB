from djitellopy import tello
import numpy as np
import cv2
import pygame

drone = tello.Tello()
cap = cv2.VideoCapture(1)   #reads the data from the camera

maxArea = 7000
minArea = 6000

maxHeight = 400
minHeight = 300

def findFace(grayImg):  #method to find the closest face
    faceCascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml") #defines the facetracking ai
    faces = faceCascade.detectmultiscale(grayImg, 1.2, 8)   #finds the face

    faceC = []  #list of all current faces
    faceA = []  #list of all areas of the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(grayImg, (x, y), (x + w, y + h), (0, 255, 0), 2)  #draws rectangle on face
        cx = x + w//2
        cy = y + h //2
        cv2.circle(grayImg, (cx, cy), 10, (0, 255, 0), cv2.FILLED)   #draws a circle in the center of the face

        area = w*h

        faceC.append[cx, cy]
        faceA.append(area)

        if len(faceA) != 0:
            i = faceA.index(max(faceA))
            return grayImg, [faceC[i], faceA[i]]    #returns the closest face
        else:
            return grayImg, [[0, 0], 0]   #if there are no faces, return 0 on everything



def trackFace(drone, info, w, pYVError, pUDError):
    area = info[1]
    x, y = info[0]

    yvError = x - w //2
    yvSpeed = 0.4 * yvError + 0.4 * (yvError - pYVError)
    yvSpeed = int(np.clip(yvSpeed, -100, 100))

    udError = y - w//2
    udSpeed = 0.2 * udError + 0.2 * (udError - pUDError)
    udSpeed = int(np.clip(udSpeed, -10, 10))

    if area < maxArea and area > minArea:
        fbSpeed = 0
    elif area > maxArea:
        fbSpeed = -20
    elif area < minArea and area != 0:
        fbSpeed = 20

    if udError < maxHeight and udError > minHeight:
        ud = 0
    elif udError > maxHeight or udError < minHeight:
        ud = udSpeed



    if x == 0:
        yvSpeed = 0
        yvError = 0
        udSpeed = 0
        udError = 0

    drone.send_rc_control(0, fbSpeed, udSpeed, yvSpeed)

    return yvError, udError

def main(drone):
    while True:
        video = drone.get_frame_read().frame
        cv2.resize(video, (360, 240))
        cv2.imshow("Camera feed", video)
        grayImg = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        cv2.resize(grayImg, (360, 240))
        grayImg, info = findFace(grayImg)
        pYVError, pUDError = trackFace(drone, info, 180, pYVError, pUDError)
    cv2.imshow("Facetracking feed", grayImg)
    cv2.waitKey(1)


if __name__ == '__main__':
    drone = tello.Tello()
    print(drone.get_battery)
    drone.streamon()
    try:
        main(drone)
    finally:
        drone.land()
