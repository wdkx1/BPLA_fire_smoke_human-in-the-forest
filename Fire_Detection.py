import cv2
import torch
from ultralytics import YOLO
import os
import json
import socket
from datetime import datetime
import math
import base64
import numpy as np

# Загрузка модели YOLO (замените на свою модель)
model = YOLO('best_yolo11.pt')

# Параметры камеры (пример, замените на реальные)
camera_params_operator = {
    "fov_horizontal": 70.0,
    "fov_vertical": 60.0,   # Угол обзора по вертикали и горизонтали
    "gps_coordinates": {
        "latitude": 55.7558,
        "longitude": 37.6173
    },
    "azimuth": 90.0,
    "elevation": 0.0
}

camera_params_uav = {
    "fov_horizontal": 70.0,
    "fov_vertical": 60.0,   # Угол обзора по вертикали и горизонтали
    "gps_coordinates": {
        "latitude": 55.7558,
        "longitude": 37.6173
    },
    "azimuth": 90.0,
    "elevation": -10.0
}

# Параметры сети для отправки данных (при необходимости раскомментировать)
TCP_IP = '127.0.0.1'
TCP_PORT = 5005

# Создание сокета и подключение к серверу (закомментировано для примера)
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((TCP_IP, TCP_PORT))

# Предварительная обработка кадра
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

    earth_radius = 6378137.0  # радиус Земли
    delta_latitude = (delta_z / earth_radius) * (180 / math.pi)
    delta_longitude = (
        delta_x / (earth_radius * math.cos(math.pi * gps_coordinates['latitude'] / 180))
    ) * (180 / math.pi)

    object_latitude = gps_coordinates['latitude'] + delta_latitude
    object_longitude = gps_coordinates['longitude'] + delta_longitude

    return {
        "latitude": object_latitude,
        "longitude": object_longitude
    }

# Основная функция анализа детекций и сбор данных в JSON
def analyze_output(results, frame, camera_id, camera_params):
    detections = []
    image_height, image_width = frame.shape[:2]

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            angle_x, angle_y = get_object_angles(
                x_pixel=object_center_x,
                y_pixel=object_center_y,
                image_width=image_width,
                image_height=image_height,
                fov_horizontal=camera_params['fov_horizontal'],
                fov_vertical=camera_params['fov_vertical']
            )

            # Абсолютный азимут и угол места объекта
            absolute_azimuth = (camera_params['azimuth'] + angle_x) % 360
            absolute_elevation = camera_params['elevation'] + angle_y

            # Пока расстояние до объекта считаем фиксированным
            distance_to_object = 100.0

            object_coordinates = calculate_object_coordinates(
                gps_coordinates=camera_params['gps_coordinates'],
                azimuth=absolute_azimuth,
                elevation=absolute_elevation,
                distance=distance_to_object
            )

            detection = {
                "timestamp": datetime.now().isoformat(),
                "camera_id": camera_id,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "gps_coordinates": camera_params['gps_coordinates'],
                "camera_direction": {
                    "azimuth": camera_params['azimuth'],
                    "elevation": camera_params['elevation']
                },
                "object_direction": {
                    "azimuth": absolute_azimuth,
                    "elevation": absolute_elevation
                },
                "object_coordinates": object_coordinates
            }

            detections.append(detection)

            # Сохранение изображения при достаточной уверенности
            if confidence > 0.6:
                label = "fire" if class_id == 0 else "smoke"
                filename = os.path.join(
                    os.getcwd(),
                    f"{label}_{camera_id}_{detection['timestamp']}.jpg"
                )
                cv2.imwrite(filename, frame)

    # Сохранение JSON на диск, если есть детекции
    if detections:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%Y-%m-%d_%H-%M")
        folder_path = os.path.join("Detections", date_str)
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{time_str}.json"
        file_path = os.path.join(folder_path, file_name)
        json_data = json.dumps({"detections": detections}, indent=4)

        try:
            with open(file_path, 'w') as f:
                json.dump(json.loads(json_data), f, indent=4)
            print(f"JSON сохранён: {file_path}")
        except Exception as e:
            print(f"Ошибка с обработкой файла: {e}")

    return detections


# Функция для отправки скриншота (с обведёнными объектами) в формате JSON
# с информацией о времени обнаружения, координатах и bounding box.
def send_screenshot_in_json(detections, frame):
    # Кодируем кадр в JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print("Ошибка кодирования кадра в JPEG.")
        return
    
    # Кодируем в base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Формируем структуру JSON
    data_to_send = {
        "timestamp": datetime.now().isoformat(),
        "message": "Object(s) detected",
        "detections": detections,
        "image_base64": encoded_image
    }

    json_string = json.dumps(data_to_send, ensure_ascii=False).encode('utf-8')

    # Пример отправки (закомментировано, чтобы не было ошибки без сервера):
    # sock.sendall(json_string)

    try:
        #  Преобразуем buffer (байты) в numpy array, а затем декодируем в изображение
        np_array = np.frombuffer(buffer, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            print("Ошибка декодирования изображения для сохранения.")
            return

        cv2.imwrite('tet.jpg', image)  # Сохраняем изображение
        print(f"Скриншот сохранен в файл: {'tet.jpg'}")
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")

        # Для демонстрации просто выведем в консоль длину сообщения
    print(f"JSON для отправки (длина {len(json_string)} байт):", data_to_send["message"])


# Инициализация видеопотоков
cap_operator = cv2.VideoCapture(0)  # Камера оператора
# cap_uav = cv2.VideoCapture(1)     # Камера БПЛА (закомментирована для примера)
#if not cap_operator.isOpened() or not cap_uav.isOpened():
#    print("Ошибка: Не удалось открыть одну или обе камеры.")
#    exit()

while True:
    ret_op, frame_op = cap_operator.read()
    if not ret_op:
        break

    # Подготовка кадра и детекция
    input_frame_op = preprocess_frame(frame_op)
    results_op = model(source=input_frame_op, save=False, verbose=False)
    detections_op = analyze_output(
        results_op,
        frame_op,
        camera_id="operator",
        camera_params=camera_params_operator
    )

    # Отрисовка bounding box на кадре, если есть обнаружения
    if detections_op:
        for det in detections_op:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame_op, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Можно также вывести текст/класс рядом с bbox (пример):
            cv2.putText(frame_op,
                        f"ID:{det['class_id']} conf:{det['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1)
        
        # Если есть детекции, отправляем скриншот в формате JSON
            send_screenshot_in_json(detections_op, frame_op)

    # Аналогичные действия для камеры БПЛА (закомментировано для примера):
    # ret_uav, frame_uav = cap_uav.read()
    # if not ret_uav:
    #     break
    # input_frame_uav = preprocess_frame(frame_uav)
    # results_uav = model(source=input_frame_uav, save=False, verbose=False)
    # detections_uav = analyze_output(results_uav, frame_uav, camera_id="uav", camera_params=camera_params_uav)
    # if detections_uav:
    #     for det in detections_uav:
    #         x1, y1, x2, y2 = det["bbox"]
    #         cv2.rectangle(frame_uav, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     send_screenshot_in_json(detections_uav, frame_uav)

    # Отображение кадра(ов) в окне
    cv2.imshow('Operator Camera', frame_op)
    # cv2.imshow('UAV Camera', frame_uav)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap_operator.release()
# cap_uav.release()
cv2.destroyAllWindows()

# Закрытие сокета, если нужно
# sock.close()
