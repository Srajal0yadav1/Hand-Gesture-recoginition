import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
import time

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
#tkinter initialoise
root= Tk()
root.title("Hand Gesture")
root.geometry("500x500")

#input for other webcam option
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
#webcam opening time 5 seconds
capture_duration = 5

#hand gesture function.
def GestFun():
    #time delay upto closing
    start_time = time.time()
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    #while True:
    while ( int(time.time() - start_time) < capture_duration ):
        # Read each frame from the webcam
        _, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
                answerlabel['text'] = f"gesture: {className}"
        # show the prediction on the frame
        # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 0, 255), 2, cv2.LINE_AA)
        print(className)
        # Show the final output
        cv2.imshow("Output", frame)

        # if cv2.waitKey(1) == ord('q'):
        #     break
    # release the webcam and destroy all active windows
    cap.release()

    cv2.destroyAllWindows()


#frame to answer
box=Frame(root,relief='sunken',padx=50,pady=50,borderwidth=3)
box.pack(pady=20)

handbutton=Button(box,text="press",font=('','20'),padx=10,command=GestFun)
handbutton.pack(pady=20)
answerlabel=Label(box,text="result",font=('','18'),pady=10,padx=10,bg='light green')
answerlabel.pack(padx=10,pady=20)

videobox=Frame(root,border=3,padx=10,pady=10,height=450)
videobox.pack(pady=20)
#mainloop tkinter

root.mainloop()