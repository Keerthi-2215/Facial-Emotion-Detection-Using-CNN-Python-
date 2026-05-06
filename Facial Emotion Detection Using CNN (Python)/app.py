import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models

# ===============================SDTY0
# CONFIG
# ===============================
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 30

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

train_dir = "dataset/train"
test_dir = "dataset/test"
model_path = "emotion_detection_model.h5"

# ===============================
# DATASET LOADING
# ===============================
dataset_exists = os.path.exists(train_dir) and os.path.exists(test_dir)

if dataset_exists:
    print("\n[INFO] Dataset found. Preparing data generators...")

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
else:
    print("\n[WARNING] Dataset not found!")
    print("Expected structure:")
    print("dataset/")
    print(" ├── train/")
    print(" └── test/")
    print("Each folder must contain emotion subfolders.")
    print("Example: Angry, Happy, Sad, etc.")

# ===============================
# CNN MODEL
# ===============================
if not os.path.exists(model_path) and dataset_exists:
    print("\n[INFO] Creating and training a new model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data
    )

    model.save(model_path)
    print(f"\n[SUCCESS] Model trained and saved as {model_path}")

elif os.path.exists(model_path):
    print(f"\n[INFO] Loading existing model from {model_path}...")
    model = models.load_model(model_path)

else:
    print("\n[ERROR] No dataset and no saved model found.")
    exit()

# ===============================
# REAL-TIME EMOTION DETECTION
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("\n[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, IMG_SIZE, IMG_SIZE, 1))

        prediction = model.predict(roi_gray, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_label = EMOTIONS[emotion_index]

        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2
        )

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
