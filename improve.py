import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import win32api
import pyautogui

# Initialize MediaPipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize variables
click_threshold = 20
click_frames = 5  # Number of consecutive frames to consider for a click
click_count = 0
smooth_factor = 0.5  # Increased smoothing factor for more stable cursor movement
prev_x, prev_y = 0, 0
scroll_mode = False
scroll_counter = 0  # To prevent accidental activation of scroll mode

# Open video capture
video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to RGB and flip horizontally
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, _ = image.shape

        # Process the image and detect hand landmarks
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        index_finger_tip = None
        thumb_tip = None
        index_finger_pip = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                # Extract landmark coordinates
                landmarks = hand_landmarks.landmark
                index_finger_tip = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    imageWidth, imageHeight)
                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_hands.HandLandmark.THUMB_TIP].x,
                    landmarks[mp_hands.HandLandmark.THUMB_TIP].y,
                    imageWidth, imageHeight)
                index_finger_pip = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    imageWidth, imageHeight)

        if index_finger_tip and thumb_tip:
            # Smooth cursor movement
            cursor_x = int(smooth_factor * index_finger_tip[0] * 4 + (1 - smooth_factor) * prev_x)
            cursor_y = int(smooth_factor * index_finger_tip[1] * 5 + (1 - smooth_factor) * prev_y)
            prev_x, prev_y = cursor_x, cursor_y

            # Move cursor
            win32api.SetCursorPos((cursor_x, cursor_y))

            # Calculate distance for clicking
            distance_click = sqrt((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2)
            if distance_click < click_threshold:
                click_count += 1
                if click_count >= click_frames:
                    print("single click")
                    pyautogui.click()
                    click_count = 0  # Reset click count after registering a click
            else:
                click_count = 0

        if index_finger_pip and index_finger_tip:
            # Calculate vertical distance for scrolling
            vertical_distance = index_finger_pip[1] - index_finger_tip[1]
            if abs(vertical_distance) > 40:  # Threshold to activate scroll mode
                scroll_counter += 1
                if scroll_counter > 5:  # Consistent gesture to activate scroll mode
                    scroll_mode = True
            else:
                scroll_counter = 0
                scroll_mode = False

            if scroll_mode:
                scroll_amount = int(vertical_distance / 5)  # Adjust the divisor to change scroll speed
                pyautogui.scroll(scroll_amount)
                if scroll_amount > 0:
                    print("scroll up")
                else:
                    print("scroll down")

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
