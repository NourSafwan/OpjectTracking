#--------------------- import the necessary libraries ---------------------#
import cv2 # Import the OpenCV library for image processing
import mediapipe as mp # Import the MediaPipe library for hand tracking
import time # Import the time library for time-related functions

#--------------------- Camera Setup ---------------------#

cap = cv2.VideoCapture(0) # Create a VideoCapture object to read from the camera

#--------------------- Hand Tracking ---------------------#

# Create an object of the Hand class from the mediapipe library
mpHands = mp.solutions.hands # Create a hands object
hands = mpHands.Hands()

# Create an object of the DrawingSpec class from the mediapipe library
mpDraw = mp.solutions.drawing_utils 

pTime = 0 # The previous time in seconds
cTime = 0 # The current time in seconds
tipIds = [4, 8, 12, 16, 20] # The ids of the tips of the fingers

right_hand = False # A boolean variable to check if the right hand is detected
left_hand = False # A boolean variable to check if the left hand is detected

#--------------------- Main Loop ---------------------#

while True: # Run an infinite loop
    success, img = cap.read() # Read the image from the camera
    img = cv2.flip(img, 1) # Flip the image horizontally
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB
    results = hands.process(imgRGB) # Process the image
    rLmList = [] # Create an empty list to store the landmarks of the right hand
    lLmList = [] # Create an empty list to store the landmarks of the left hand

    #--------------------- Hand Landmarks ---------------------#
    if results.multi_hand_landmarks: # If there are hands in the image
        for handLms in results.multi_hand_landmarks: # For each hand
            for id, lm in enumerate(handLms.landmark): # For each landmark
                h, w, c = img.shape # Get the height, width, and channels of the image
                cx, cy = int(lm.x * w), int(lm.y * h) # Get the x and y coordinates of the landmark
                
                #--------------------- Right or Left Hand ---------------------#
                #check if this is the right hand or left hand by checking the x coordinate of the tip of the thumb
                if id == 4:
                    if lm.x < handLms.landmark[17].x: # If the tip of the thumb is to the left of the base of the pinky
                        right_hand = True # Set the right hand to True
                        left_hand =False
                    else:
                        left_hand = True # Set the left hand to True
                        right_hand =False

                #--------------------- Landmark Visualization ---------------------#
                if id == 0: # If the landmark is the first landmark
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # Draw a circle around the landmark
                if id == 4 or id == 8: # If the landmark is the 4 or 8 landmark
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) # Draw a green circle around the landmark
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Draw the landmarks and connections

                #--------------------- Right Hand ---------------------#
                if right_hand: # If the right hand is detected
                    cv2.putText(img,"Right Hand", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2) # Display "Right Hand"
                    
                    rLmList.append([id, cx, cy]) # Append the id, x, and y coordinates to the list of landmarks
                
                    if len(rLmList) == 21: # If all landmarks are detected
                        fingers = [] # Create an empty list to store the fingers
                        if rLmList[4][1] < rLmList[3][1]: # If the tip of the thumb is to the right of the base of the thumb (finger is open)
                            fingers.append(1)
                        else:
                            fingers.append(0) 
                        for id in range(1, 5): # For each finger (index, middle, ring, pinky)
                            if rLmList[tipIds[id]][2] < rLmList[tipIds[id] - 2][2]: # If the tip of the finger is above the base of the finger
                                fingers.append(1)
                            else:
                                fingers.append(0)
                        totalFingers = fingers.count(1) # Count the number of fingers that are open
                        #--------------------- Display the Number of Fingers ---------------------#
                        cv2.putText(img,"Opened fingers: " + str(totalFingers), (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                
                
                #--------------------- Left Hand ---------------------#
                if left_hand: # If the left hand is detected
                    cv2.putText(img,"Left Hand", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    lLmList.append([id, cx, cy]) # Append the id, x, and y coordinates to the list of landmarks
                    if len(lLmList) == 21: # If all landmarks are detected
                        fingers = []
                        if lLmList[4][1] > lLmList[3][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                        for id in range(1, 5):
                            if lLmList[tipIds[id]][2] < lLmList[tipIds[id] - 2][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                        totalFingers = fingers.count(1)
                        cv2.putText(img,"Opened fingers: " + str(totalFingers), (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    #--------------------- Frame Rate ---------------------#
    cTime = time.time() # Get the current time in seconds
    fps = 1 / (cTime - pTime) # Calculate the frames per second
    pTime = cTime # Set the previous time to the current time

    cv2.putText(img,"fps: " +  str(int(fps)), (10, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2) # Display the frames per second

    #--------------------- Display the Image ---------------------#
    cv2.imshow("Hand Tracker", img) # Display the image
    if cv2.waitKey(5) & 0xff == 27: # If the escape key is pressed
        cv2.destroyAllWindows() # Close the window
        break # Break the loop