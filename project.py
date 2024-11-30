import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('drowsiness_detection_model.h5')

labels = ['Drowsy', 'Awake']

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(gray, (64, 64))

    img = img / 255.0

    img = np.expand_dims(img, axis=-1)

    img = np.expand_dims(img, axis=0)

    return img

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess_frame(frame)

    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]

    cv2.putText(frame, f'Status: {label}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
