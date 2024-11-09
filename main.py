import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime

# Загрузка модели YOLOv10
model = YOLO('best.pt')


# Захват видеопотока и прогнозирование
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    return frame


# Анализ детекций для поиска огня и дыма
def analyze_output(results, frame):
    label = ""
    max_confidence = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf.item()

            if confidence > max_confidence:
                max_confidence = confidence
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if class_id == 0 and confidence > 0.3:
                    label = f"Пожар обнаружен с вероятностью {confidence:.2f}"
                    if confidence > 0.6:
                        filename = os.path.join(os.getcwd(), f"fire_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
                elif class_id == 1 and confidence > 0.5:
                    label = f"Обнаружен дым с вероятностью {confidence:.2f}"
                    if confidence > 0.6:
                        filename = os.path.join(os.getcwd(), "smoke.jpg")
                        cv2.imwrite(filename, frame)

    if label:
        print(label)
    return label


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = preprocess_frame(frame)
    results = model(source=input_frame, save=False, verbose=False)

    # Анализ детекций
    label = analyze_output(results, frame)

    if label:
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()