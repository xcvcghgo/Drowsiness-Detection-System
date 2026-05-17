from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import os
import time

app = Flask(**name**, static_folder=“templates”)
CORS(app)

mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)

# ===== نقاط العين (6 نقاط لكل عين) =====

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 81, 13, 311, 308, 178]

FACE_OUTLINE  = [10,338,297,332,284,251,389,356,454,323,361,288,
397,365,379,378,400,377,152,148,176,149,150,136,
172,58,132,93,234,127,162,21,54,103,67,109,10]
EYE_CONTOUR_L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
EYE_CONTOUR_R = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
MOUTH_CONTOUR = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]

# ===== الإعدادات =====

EAR_THRESHOLD_RATIO = 0.78   # نسبة من baseline الشخص (معايرة تلقائية)
MAR_THRESHOLD       = 0.65   # عتبة التثاوب
YAWN_FRAMES         = 10     # عدد frames للتأكيد على التثاوب
CALIBRATION_FRAMES  = 45     # عدد frames للمعايرة (~3 ثواني)
EAR_SMOOTH_FRAMES   = 7      # عدد frames للتمهيد

# ===== متغيرات التمهيد والمعايرة =====

ear_history      = deque(maxlen=EAR_SMOOTH_FRAMES)
calibration_data = {“frames”: 0, “ear_sum”: 0.0, “done”: False}
EAR_THRESHOLD    = 0.20  # قيمة افتراضية حتى تنتهي المعايرة

# ===== حالة النظام =====

current_data = {
“eye”: 0, “jaw”: 0,
“drowsy”: 0, “yawn”: 0,
“sleep_count”: 0, “yawn_count”: 0,
“alert_type”: None,
“landmarks”: {},
“calibrated”: False,
“ear_threshold”: 20,
“calibration_progress”: 0
}

yawn_frames           = 0
is_yawning            = False
eye_closed_start_time = None
is_drowsy_triggered   = False

def calc_ratio(points):
A = dist.euclidean(points[1], points[5])
B = dist.euclidean(points[2], points[4])
C = dist.euclidean(points[0], points[3])
return (A + B) / (2.0 * C) if C != 0 else 0

@app.route(”/”)
def home():
return send_from_directory(“templates”, “index.html”)

@app.route(”/api/process”, methods=[“POST”])
def process():
global yawn_frames, current_data, is_yawning
global eye_closed_start_time, is_drowsy_triggered
global EAR_THRESHOLD, calibration_data, ear_history

```
file  = request.files["frame"]
npimg = np.frombuffer(file.read(), np.uint8)
frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

h, w, _ = frame.shape
rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)

eye_val       = 0
jaw_val       = 0
drowsy        = 0
yawn          = 0
alert         = None
landmarks_out = {}

if results.multi_face_landmarks:
    lm = results.multi_face_landmarks[0].landmark

    left_pts  = np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT_EYE])
    right_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT_EYE])
    mouth_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in MOUTH])

    ear_raw = (calc_ratio(left_pts) + calc_ratio(right_pts)) / 2
    mar_val = calc_ratio(mouth_pts)

    # ===== معايرة تلقائية لكل مستخدم =====
    if not calibration_data["done"]:
        calibration_data["ear_sum"] += ear_raw
        calibration_data["frames"]  += 1
        progress = int((calibration_data["frames"] / CALIBRATION_FRAMES) * 100)
        current_data["calibration_progress"] = min(progress, 100)

        if calibration_data["frames"] >= CALIBRATION_FRAMES:
            baseline      = calibration_data["ear_sum"] / calibration_data["frames"]
            EAR_THRESHOLD = baseline * EAR_THRESHOLD_RATIO
            calibration_data["done"]          = True
            current_data["calibrated"]        = True
            current_data["ear_threshold"]     = int(EAR_THRESHOLD * 100)
            current_data["calibration_progress"] = 100

    # ===== تمهيد القراءة (يمنع التذبذب) =====
    ear_history.append(ear_raw)
    ear_smooth = sum(ear_history) / len(ear_history)

    eye_val = int(ear_smooth * 100)
    jaw_val = int(mar_val * 100)

    # ===== كشف النعاس =====
    if ear_smooth < EAR_THRESHOLD:
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        elapsed = time.time() - eye_closed_start_time
        if elapsed >= 6.0:
            alert = "emergency"
        else:
            alert = "warning"
            if not is_drowsy_triggered:
                current_data["sleep_count"] += 1
                drowsy = 1
                is_drowsy_triggered = True
    else:
        alert = "clear" if eye_closed_start_time else None
        eye_closed_start_time = None
        is_drowsy_triggered   = False

    # ===== كشف التثاوب =====
    if mar_val > MAR_THRESHOLD:
        yawn_frames += 1
        if yawn_frames >= YAWN_FRAMES:
            if not is_yawning:
                is_yawning = True
                current_data["yawn_count"] += 1
                yawn = 1
            if alert is None:
                alert = "yawn"
    else:
        yawn_frames = 0
        is_yawning  = False

    def pts(indices):
        return [[round(lm[i].x, 4), round(lm[i].y, 4)] for i in indices]

    landmarks_out = {
        "face":  pts(FACE_OUTLINE),
        "left":  pts(EYE_CONTOUR_L),
        "right": pts(EYE_CONTOUR_R),
        "mouth": pts(MOUTH_CONTOUR)
    }

else:
    if eye_closed_start_time is not None:
        alert = "clear"
    eye_closed_start_time = None
    is_drowsy_triggered   = False
    yawn_frames           = 0
    is_yawning            = False
    ear_history.clear()

current_data.update({
    "eye": eye_val, "jaw": jaw_val,
    "drowsy": drowsy, "yawn": yawn,
    "alert_type": alert,
    "landmarks": landmarks_out
})

return jsonify(current_data)
```

@app.route(”/api/stop”, methods=[“POST”])
def stop_system():
global eye_closed_start_time, is_drowsy_triggered, yawn_frames
global is_yawning, current_data, calibration_data, EAR_THRESHOLD, ear_history

```
eye_closed_start_time = None
is_drowsy_triggered   = False
yawn_frames           = 0
is_yawning            = False
ear_history.clear()

# إعادة تعيين المعايرة عند الإيقاف
calibration_data = {"frames": 0, "ear_sum": 0.0, "done": False}
EAR_THRESHOLD    = 0.20

current_data.update({
    "eye": 0, "jaw": 0,
    "drowsy": 0, "yawn": 0,
    "sleep_count": 0, "yawn_count": 0,
    "alert_type": "clear",
    "landmarks": {},
    "calibrated": False,
    "ear_threshold": 20,
    "calibration_progress": 0
})
return jsonify(current_data)
```

@app.route(”/api/data”)
def data():
return jsonify(current_data)

if **name** == “**main**”:
app.run(host=“0.0.0.0”, port=int(os.environ.get(“PORT”, 5000)), debug=True)
