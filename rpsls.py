"""
DOCSTRING

Todo:
    Add angles between fingers
"""

import time
import typing as t
from enum import auto

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from strenum import StrEnum  # pip install StrEnum

# ------------------------------------------------------------------------------
# Rock, paper, scissor, lizard, spock


class Gesture(StrEnum):
    rock = auto()
    paper = auto()
    scissors = auto()
    lizard = auto()
    spock = auto()


_VERB = {
    (Gesture.rock, Gesture.scissors): "crushes",
    (Gesture.rock, Gesture.lizard): "crushes",
    (Gesture.paper, Gesture.rock): "covers",
    (Gesture.paper, Gesture.spock): "disproves",
    (Gesture.scissors, Gesture.paper): "cuts",
    (Gesture.scissors, Gesture.lizard): "decapitates",
    (Gesture.lizard, Gesture.paper): "eats",
    (Gesture.lizard, Gesture.spock): "poisons",
    (Gesture.spock, Gesture.rock): "vaporizes",
    (Gesture.spock, Gesture.scissors): "smashes",
}


def evaluate(gesture1: Gesture, gesture2: Gesture) -> t.Tuple[t.Optional[int], str]:
    """Return winner index and text"""

    if (gesture1, gesture2) in _VERB:
        return 0, f"{gesture1} {_VERB[gesture1, gesture2]} {gesture2}"
    if (gesture2, gesture1) in _VERB:
        return 1, f"{gesture2} {_VERB[gesture2, gesture1]} {gesture1}"
    return None, "Tie"


# ------------------------------------------------------------------------------
# Open CV

CAMERA = 0

PREV_TIME = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = 255, 255, 255
GREEN = 0, 255, 0
RED = 255, 0, 0


def put_text(image, text, org, color, font_face=FONT, font_scale=2, thickness=3):
    """DOCSTRING"""
    cv2.putText(
        image,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )


def frame_rate(image, pt=(10, 70), color=GREEN):
    """DOCSTRING"""
    global PREV_TIME
    curr_time = time.time()
    fps = 1 // (curr_time - PREV_TIME)
    PREV_TIME = curr_time
    put_text(image, str(fps), pt, color)


def image_generator():
    """DOCSTRING"""
    cap = cv2.VideoCapture(CAMERA)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            continue

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image


# ------------------------------------------------------------------------------
# Machine Learning
class KNN:
    def __init__(self, k: int, dataset_path: str, response_dict=None):
        # Gesture recognition model
        csv_file = np.genfromtxt(dataset_path, delimiter=",")
        angles = csv_file[:, 1:].astype(np.float32)
        responses = csv_file[:, 0].astype(np.float32)

        self.k = k
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(angles, cv2.ml.ROW_SAMPLE, responses)
        self.response_dict = response_dict

    def find_nearest(self, data):
        """DOCSTRING"""
        # ret, results, neighbours, dist
        _, results, _, _ = self.knn.findNearest(data, self.k)
        response = int(results[0][0])
        if self.response_dict:
            return self.response_dict.get(response)
        return response


# ------------------------------------------------------------------------------
# Hands

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def joint_angles(landmark):
    """DOCSTRING"""
    joint = np.array([(lm.x, lm.y, lm.z) for lm in landmark])

    # Compute angles between joints

    # thumb, index, middle, ring, pinky
    parent_index = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    child_index_ = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]
    segment1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
    segment2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    # Angles between fingers
    # segment1 += [0, 4, 8, 12]
    # segment2 += [4, 8, 12, 16]

    v = joint[child_index_, :] - joint[parent_index, :]

    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using acos of dot product
    angle = np.degrees(np.arccos(np.einsum("nt,nt->n", v[segment1, :], v[segment2, :])))

    # Inference gesture
    data = np.array([angle], dtype=np.float32)
    return data


class HandDetector:  # KNN
    """DOCSTRING"""

    def __init__(
        self,
        knn,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.joint_spec = mp_draw.DrawingSpec(color=RED, thickness=2, circle_radius=6)
        self.finger_spec = mp_draw.DrawingSpec(
            color=WHITE, thickness=2, circle_radius=2
        )

        self.knn = knn

    def find_gesture(self, landmark):
        """DOCSTRING"""
        data = joint_angles(landmark)
        return self.knn.find_nearest(data)

    def draw_hand(self, image, hand_result):
        """DOCSTRING"""
        mp_draw.draw_landmarks(
            image,
            hand_result,
            mp_hands.HAND_CONNECTIONS,
            self.joint_spec,
            self.finger_spec,
        )

    def find_hands(self, image):
        """DOCSTRING"""

        hand_results = self.hands.process(image).multi_hand_landmarks
        if hand_results is None:
            return image, []

        height, width = image.shape[:2]

        gestures = []
        for hand_result in hand_results:
            self.draw_hand(image, hand_result)

            gesture = self.find_gesture(hand_result.landmark)
            if gesture is None:
                continue

            pt = hand_result.landmark[0]
            pt = (
                int(pt.x * width),
                int(pt.y * height) + 20,
            )
            gestures.append((gesture, pt))

        return image, gestures


# ------------------------------------------------------------------------------
# Hands

DATASET_NAME = "dataset.csv"
RPS_GESTURE = {
    1: Gesture.rock,
    2: Gesture.paper,
    3: Gesture.scissors,
    4: Gesture.lizard,
    5: Gesture.spock,
}


def rpsls() -> None:
    """DOCSTRING"""
    global CAMERA

    # Streamlit
    st.title("Rock Paper Scissor Spock Lizard")
    CAMERA = st.number_input("Camera", min_value=0, max_value=4, value=0, step=1)
    frame = st.image([])
    result = st.empty()

    knn = KNN(5, DATASET_NAME, RPS_GESTURE)

    detector = HandDetector(knn)

    for image in image_generator():
        image, gestures = detector.find_hands(image)

        for gesture, pt in gestures:
            put_text(image, gesture.upper(), pt, WHITE, font_scale=1, thickness=2)

        text = ""
        if len(gestures) >= 2:
            winner, text = evaluate(gestures[0][0], gestures[1][0])

            if winner is not None:
                x, y = gestures[winner][1]
                put_text(image, "Winner", (x, y + 70), GREEN)
                # cv2.imwrite("example.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        frame.image(image)
        result.markdown(text)

    result.markdown("Stopped")


if __name__ == "__main__":
    rpsls()
