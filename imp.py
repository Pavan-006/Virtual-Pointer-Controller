import mediapipe as mp
import cv2
import numpy as np
from math import sqrt
import pyautogui
import time  # Added for delay handling

# Initialize MediaPipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize variables
click_threshold = 20  # Distance threshold for a click gesture
drag_threshold = 15  # Distance threshold for a drag gesture
click_frames = 5  # Number of consecutive frames to register a click
click_count = 0
smooth_factor = 0.9  # Smoothing factor for cursor movement
prev_x, prev_y = 0, 0
scroll_mode = False
scroll_counter = 0
drag_mode = False
drag_start_counter = 0
last_click_time = 0
click_delay = 0.25  # Minimum delay between clicks in seconds
pick_mode = False  # Indicates whether pick mode is active

# Sensitivity settings
cursor_speed = 2.5  # Adjusted to make cursor move faster or slower
scroll_speed = 8    # Scroll speed sensitivity

# Open video capture
video = cv2.VideoCapture(0)

# Define function for dynamic smoothing
def dynamic_smoothing(current, previous, factor):
    return int(factor * current + (1 - factor) * previous)

# Define function to calculate the distance between two points
def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Define function for checking if landmarks are valid
def are_landmarks_valid(*landmarks):
    return all(landmark is not None for landmark in landmarks)

# Function to draw circles at landmarks for better visual feedback
def draw_landmark_circles(image, landmarks, color=(0, 255, 0), radius=5):
    for landmark in landmarks:
        if landmark is not None:
            cv2.circle(image, landmark, radius, color, cv2.FILLED)

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

        # Draw circles on the landmarks for better visual feedback
        draw_landmark_circles(image, [index_finger_tip, thumb_tip, index_finger_pip], color=(0, 255, 0), radius=5)

        # Check if all required landmarks are valid
        if are_landmarks_valid(index_finger_tip, thumb_tip):
            # Smooth cursor movement with dynamic smoothing
            cursor_x = dynamic_smoothing(index_finger_tip[0] * cursor_speed, prev_x, smooth_factor)
            cursor_y = dynamic_smoothing(index_finger_tip[1] * cursor_speed, prev_y, smooth_factor)
            prev_x, prev_y = cursor_x, cursor_y

            # Move cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Calculate distance for clicking and dragging
            distance_click = calculate_distance(index_finger_tip, thumb_tip)

            # Drag gesture detection
            if distance_click < drag_threshold:
                drag_start_counter += 1
                if drag_start_counter >= click_frames:  # Consistent pinch to start drag
                    if not drag_mode:
                        pyautogui.mouseDown()
                        drag_mode = True
                        pick_mode = True  # Activate pick mode
                        print("Drag started")
                        # Provide visual feedback for pick mode
                        cv2.putText(image, 'Picking', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                drag_start_counter = 0
                if drag_mode:
                    pyautogui.mouseUp()
                    drag_mode = False
                    pick_mode = False  # Deactivate pick mode
                    print("Drag ended")
                    # Provide visual feedback for drop mode
                    cv2.putText(image, 'Dropped', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Click gesture detection
            current_time = time.time()
            if not drag_mode and (current_time - last_click_time) > click_delay:  # Only check for clicks if not dragging
                if distance_click < click_threshold:
                    click_count += 1
                    if click_count >= click_frames:
                        print("Single click")
                        pyautogui.click()
                        last_click_time = current_time
                        click_count = 0  # Reset click count after registering a click
                else:
                    click_count = 0

        # Check if index_finger_pip and index_finger_tip are valid for scrolling
        if are_landmarks_valid(index_finger_pip, index_finger_tip):
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
                scroll_amount = int(vertical_distance / scroll_speed)  # Adjust the divisor to change scroll speed
                pyautogui.scroll(scroll_amount)
                if scroll_amount > 0:
                    print("Scroll up")
                else:
                    print("Scroll down")

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
