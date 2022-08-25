#!/usr/bin/env python

"""
Update dataset for a single gesture
"""

from collections import Counter

import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

from rpsls import RPS_GESTURE, DATASET_NAME, joint_angles, image_generator

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dataframe(data):
    gestures = Counter(data[:, 0])
    a = [(RPS_GESTURE.get(int(gesture_id), str(int(gesture_id))).title(), count)
         for gesture_id, count in gestures.items()]
    df = pd.DataFrame(np.array(a), columns=('Gesture', 'Count'))

    '''
    #Analyze
    print("  ", end="")
    for gid in gestures:
        print(f"    {RPS_GESTURE[int(gid)]:7}", end="")
    print()
    for i in range(1, 16):
        print(f"{i:2}", end="")
        for gid in gestures:
            index = (data[:, 0] == gid)
            values = data[index, i]
            lower = int(values.min())
            upper = int(values.max())
            print(f"     {lower:3} {upper:3}", end="")
        print()
    '''

    return df


def train():
    """TODO"""
    # Gesture recognition data
    global data, entry
    entry = None
    data = np.genfromtxt(DATASET_NAME, delimiter=",")

    def add_gesture():
        global data, entry
        if entry is None:
            return
        data = np.vstack((data, entry))
        np.savetxt(DATASET_NAME, data, delimiter=",", fmt = '%9f')
        entry = None

    st.title("Train Hands")
    gesture_number = st.number_input('Gesture number',
              value=6, min_value=0, step=1, format="%d")
    st.button("Add", key="a", help="Add this gesture to the dataset",
              on_click=add_gesture)
    frame = st.image([])
    st.table(dataframe(data))

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for image in image_generator():
        hand_results = hands.process(image).multi_hand_landmarks
        if hand_results is None:
            continue

        for hand_result in hand_results:
            mp_drawing.draw_landmarks(image, hand_result, mp_hands.HAND_CONNECTIONS)
            entry = np.append(gesture_number, joint_angles(hand_result.landmark))

        frame.image(image)

    else:
        st.write("Stopped")


if __name__ == "__main__":
    # streamlit run train.py
    train()
