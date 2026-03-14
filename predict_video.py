import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from collections import Counter

# load model

model = joblib.load("xgb_gesture_model_5.pkl")
scaler = joblib.load("scaler_5.pkl")
feature_names = joblib.load("feature_names_5.pkl")

LABELS_INV = {
    0: "tripping",
    1: "high_sticking",
    2: "holding",
    3: "cross_checking",
    4: "interference"
}

WINDOW_SIZE = 10
STEP = 5

# mediapipe

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# pose functions

def extract_pose(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    data = []

    for lm in result.pose_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(data)


def get_point(p, idx):
    return p[idx*4: idx*4+2]


def angle(a, b, c):

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)

    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# frame features

def frame_features(p):

    ls = get_point(p, 11)
    le = get_point(p, 13)
    lw = get_point(p, 15)

    rs = get_point(p, 12)
    re = get_point(p, 14)
    rw = get_point(p, 16)

    nose = get_point(p, 0)

    shoulder_dist = np.linalg.norm(ls - rs)

    vertical = np.array([0, -1])

    return {

        "left_elbow_angle": angle(ls, le, lw),
        "right_elbow_angle": angle(rs, re, rw),

        "left_wrist_x": lw[0],
        "left_wrist_y": lw[1],

        "right_wrist_x": rw[0],
        "right_wrist_y": rw[1],

        "left_wrist_above_head": int(lw[1] < nose[1]),
        "right_wrist_above_head": int(rw[1] < nose[1]),

        "one_hand_high": int((lw[1] < nose[1]) ^ (rw[1] < nose[1])),

        "hands_distance": np.linalg.norm(lw - rw),

        "hands_x_diff": abs(lw[0] - rw[0]),
        "hands_y_diff": abs(lw[1] - rw[1]),

        "wrists_angle": np.degrees(np.arctan2(lw[1]-rw[1], lw[0]-rw[0])),

        "left_wrist_vs_shoulder": lw[1] - ls[1],
        "right_wrist_vs_shoulder": rw[1] - rs[1],

        "left_arm_length": np.linalg.norm(lw - ls),
        "right_arm_length": np.linalg.norm(rw - rs),

        "left_arm_angle": angle(le, ls, rs),
        "right_arm_angle": angle(re, rs, ls),

        "norm_left_wrist_x": (lw[0] - ls[0]) / (shoulder_dist + 1e-6),
        "norm_left_wrist_y": (lw[1] - ls[1]) / (shoulder_dist + 1e-6),

        "norm_right_wrist_x": (rw[0] - rs[0]) / (shoulder_dist + 1e-6),
        "norm_right_wrist_y": (rw[1] - rs[1]) / (shoulder_dist + 1e-6),

        "left_wrist_vs_head": nose[1] - lw[1],
        "right_wrist_vs_head": nose[1] - rw[1],

        "hands_height_diff": abs(lw[1] - rw[1]),

        "left_arm_vertical_angle": angle(ls + vertical, ls, lw),
        "right_arm_vertical_angle": angle(rs + vertical, rs, rw)
    }


# window features

def window_features(df):

    features = {}

    for col in df.columns:

        features[f"mean_{col}"] = df[col].mean()
        features[f"max_{col}"] = df[col].max()
        features[f"std_{col}"] = df[col].std()

    features["range_left_wrist_y"] = df["left_wrist_y"].max() - df["left_wrist_y"].min()
    features["range_right_wrist_y"] = df["right_wrist_y"].max() - df["right_wrist_y"].min()

    motion = (
        df["lw_speed_x"] + df["rw_speed_x"] +
        df["lw_speed_y"] + df["rw_speed_y"]
    )

    horizontal = df["lw_speed_x"] + df["rw_speed_x"]
    vertical = df["lw_speed_y"] + df["rw_speed_y"]

    features["active_motion_ratio"] = (motion > motion.mean()).mean()
    features["motion_orientation_ratio"] = horizontal.mean() / (vertical.mean() + 1e-6)

    features["ratio_hand_above_head"] = (
        df["left_wrist_above_head"].sum() +
        df["right_wrist_above_head"].sum()
    ) / (2 * len(df))

    return features


# predict video

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        pose_vec = extract_pose(frame)

        if pose_vec is not None:
            frames.append(frame_features(pose_vec))

    cap.release()

    if len(frames) < WINDOW_SIZE:
        return "Unknown", 0.0

    df = pd.DataFrame(frames)

    df["lw_speed_x"] = df["left_wrist_x"].diff().abs()
    df["rw_speed_x"] = df["right_wrist_x"].diff().abs()

    df["lw_speed_y"] = df["left_wrist_y"].diff().abs()
    df["rw_speed_y"] = df["right_wrist_y"].diff().abs()

    df["le_speed"] = df["left_elbow_angle"].diff().abs()
    df["re_speed"] = df["right_elbow_angle"].diff().abs()

    predictions = []
    confidences = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP):

        window = df.iloc[start:start+WINDOW_SIZE]

        features = window_features(window)

        X = pd.DataFrame([features])

        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0

        X = X[feature_names]

        X_scaled = scaler.transform(X)

        probs = model.predict_proba(X_scaled)[0]

        pred = np.argmax(probs)
        confidence = np.max(probs)

        predictions.append(pred)
        confidences.append(confidence)

    if not predictions:
        return "Unknown", 0.0

    final_pred = Counter(predictions).most_common(1)[0][0]
    avg_conf = float(np.mean(confidences))

    return LABELS_INV[final_pred], avg_conf


# test

if __name__ == "__main__":

    video = r"test_videos\tripping\tripping.MP4"

    gesture, confidence = predict_video(video)

    print("Predicted gesture:", gesture)
    print("Confidence:", round(confidence*100, 2), "%")
