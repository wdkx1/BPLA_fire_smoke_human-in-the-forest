import cv2
import torch
from ultralytics import YOLO
import os
import json
import math
from datetime import datetime

# Загрузка модели YOLO
model = YOLO('best_yolo11.pt')

# Функция для предварительной обработки изображения
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    return frame

# Вычисление угловых отклонений объекта на изображении
def get_object_angles(x_pixel, y_pixel, image_width, image_height, fov_horizontal, fov_vertical):
    center_x = image_width / 2
    center_y = image_height / 2
    delta_x = x_pixel - center_x
    delta_y = center_y - y_pixel
    angle_x = (delta_x / center_x) * (fov_horizontal / 2)
    angle_y = (delta_y / center_y) * (fov_vertical / 2)
    return angle_x, angle_y

# Функция для вычисления координат объекта


def calculate_object_coordinates(gps_coordinates, azimuth, elevation, distance):
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    delta_x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    delta_y = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    delta_z = distance * math.sin(elevation_rad)

    earth_radius = 6378137.0
    delta_latitude = (delta_z / earth_radius) * (180 / math.pi)
    delta_longitude = (delta_x / (earth_radius * math.cos(math.pi * gps_coordinates['latitude'] / 180))) * (180 / math.pi)

    object_latitude = gps_coordinates['latitude'] + delta_latitude
    object_longitude = gps_coordinates['longitude'] + delta_longitude

    return {"latitude": object_latitude, "longitude": object_longitude}

# Функция анализа детекций
def analyze_detections(detections, frame, gps_coordinates, azimuth, fov_horizontal, fov_vertical):
    height, width, _ = frame.shape
    results = []

    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
        
        # Вычисляем центральные координаты объекта
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Вычисляем углы отклонения
        angle_x, angle_y = get_object_angles(x_center, y_center, width, height, fov_horizontal, fov_vertical)

        # Координаты GPS объекта (пока что без данных о расстоянии)
        object_coords = calculate_object_coordinates(gps_coordinates, azimuth, angle_y, distance=0)

        # Сохраняем объект в список результатов
        results.append({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "angles": {"x": angle_x, "y": angle_y},
            "gps": object_coords
        })

        # Рисуем прямоугольник вокруг объекта и добавляем текст
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"Class: {int(class_id)}, Conf: {confidence:.2f}"
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Делаем скриншот и сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_name = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(screenshot_name, frame)
        print(f"Скриншот сохранён: {screenshot_name}")

    return results

# Основной процесс
def main(video_source, gps_coordinates, azimuth, fov_horizontal=90, fov_vertical=60):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)

        # YOLO предсказания
        results = model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()

        # Анализ детекций
        analyze_detections(detections, frame, gps_coordinates, azimuth, fov_horizontal, fov_vertical)

        # Отображение результата
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Пример GPS координат
gps_coordinates = {"latitude": 55.7558, "longitude": 37.6173}

# Запуск программы
if __name__ == "__main__":
    video_source = 0  # Камера по умолчанию
    azimuth = 45  # Текущий азимут камеры
    main(video_source, gps_coordinates, azimuth)
