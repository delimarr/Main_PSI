#Simon Eich
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from globals import camera_index

def camera_video():
    # Capture video from camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error opening video capture device.")
        exit()

    # Create a window to display the video
    cv2.namedWindow("ELP Camera Feed", cv2.WINDOW_NORMAL)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame capture was successful
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        # Display the resulting frame
        cv2.imshow("ELP Camera Feed", frame)

        # Capture photo on button press
        key = cv2.waitKey(1)
        # Exit on 'Esc' key press (ASCII code 27)
        if key ==27:
            break
        
        elif key != -1:
            # If any other key is pressed, ignore it
            continue

    # When everything is done, release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()
