import sys
import cv2
import os
import json
import math
from datetime import datetime
from ultralytics import YOLO
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QGroupBox, QGridLayout

# Загрузка модели YOLO
model = YOLO('best_yolo11.pt')

# Параметры камеры
camera_params_operator = {
    "fov_horizontal": 70.0,
    "fov_vertical": 60.0,  
    "gps_coordinates": {
        "latitude": 55.7558,
        "longitude": 37.6173
    },
    "azimuth": 90.0,
    "elevation": 0.0
}

# Обработчик интерфейса для отображения видео
class VideoWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кадры с камеры")
        self.setGeometry(100, 100, 640, 480)

        layout = QVBoxLayout()
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def update_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))


# Обработчик интерфейса для отображения информации о детекциях
class InfoWindow(QtWidgets.QWidget):
    update_data_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерфейс Детекции Объектов")
        self.setGeometry(750, 100, 500, 600)

        layout = QVBoxLayout()
        layout.addWidget(self.create_info_block())
        self.setLayout(layout)
        self.update_data_signal.connect(self.update_ui)

    def create_info_block(self):
        group = QGroupBox("Детекция объекта")
        layout = QGridLayout()

        self.label_class = QLabel("Класс объекта: ---")
        self.label_confidence = QLabel("Уверенность: ---")
        self.label_coords = QLabel("GPS Координаты: ---")
        self.label_azimuth = QLabel("Азимут: ---")
        self.label_elevation = QLabel("Угол места: ---")
        self.label_time = QLabel("Время детекции: ---")

        layout.addWidget(self.label_class, 0, 0)
        layout.addWidget(self.label_confidence, 1, 0)
        layout.addWidget(self.label_coords, 2, 0)
        layout.addWidget(self.label_azimuth, 3, 0)
        layout.addWidget(self.label_elevation, 4, 0)
        layout.addWidget(self.label_time, 5, 0)

        group.setLayout(layout)
        return group

    def update_ui(self, data):
        self.label_class.setText(f"Класс объекта: {data['class_id']}")
        self.label_confidence.setText(f"Уверенность: {data['confidence']:.2f}")
        self.label_coords.setText(f"GPS Координаты: {data['gps_coordinates']['latitude']:.6f}, {data['gps_coordinates']['longitude']:.6f}")
        self.label_azimuth.setText(f"Азимут: {data['azimuth']:.2f}")
        self.label_elevation.setText(f"Угол места: {data['elevation']:.2f}")
        self.label_time.setText(f"Время детекции: {data['timestamp']}")

    def analyze_output(self, results, frame, camera_params):
        detections = []
        image_height, image_width = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                object_center_x = (x1 + x2) / 2
                object_center_y = (y1 + y2) / 2

                angle_x, angle_y = self.get_object_angles(
                    x_pixel=object_center_x,
                    y_pixel=object_center_y,
                    image_width=image_width,
                    image_height=image_height,
                    fov_horizontal=camera_params['fov_horizontal'],
                    fov_vertical=camera_params['fov_vertical']
                )

                absolute_azimuth = (camera_params['azimuth'] + angle_x) % 360
                absolute_elevation = camera_params['elevation'] + angle_y

                distance_to_object = 100.0
                object_coordinates = self.calculate_object_coordinates(
                    gps_coordinates=camera_params['gps_coordinates'],
                    azimuth=absolute_azimuth,
                    elevation=absolute_elevation,
                    distance=distance_to_object
                )

                detection = {
                    "class_id": class_id,
                    "confidence": confidence,
                    "gps_coordinates": object_coordinates,
                    "azimuth": absolute_azimuth,
                    "elevation": absolute_elevation,
                    "timestamp": datetime.now().isoformat(),
                    "bbox": [x1, y1, x2, y2]
                }

                detections.append(detection)
        
        return detections

    def get_object_angles(self, x_pixel, y_pixel, image_width, image_height, fov_horizontal, fov_vertical):
        center_x = image_width / 2
        center_y = image_height / 2
        delta_x = x_pixel - center_x
        delta_y = center_y - y_pixel

        angle_x = (delta_x / center_x) * (fov_horizontal / 2)
        angle_y = (delta_y / center_y) * (fov_vertical / 2)
        return angle_x, angle_y

    def calculate_object_coordinates(self, gps_coordinates, azimuth, elevation, distance):
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

    def update_data(self, frame):
        input_frame_op = frame  # Используем оригинальный кадр

        # Детекция объектов
        results_op = model(source=input_frame_op, save=False, verbose=False)

        # Обработка результатов детекции
        detections_op = self.analyze_output(results_op, frame, camera_params_operator)

        # Если есть детекции, обновляем интерфейс
        if detections_op:
            self.update_data_signal.emit(detections_op[0])  # Отправляем данные на интерфейс

        for det in detections_op:
            if 'bbox' in det:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(frame,
                            f"ID:{det['class_id']} conf:{det['confidence']:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1)

        # Обновляем изображение с камеры в VideoWindow
        video_window.update_image(frame)  # Важно вызвать update_image на video_window, а не info_window

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Создаём оба окна
    video_window = VideoWindow()
    info_window = InfoWindow()
    
    video_window.show()
    info_window.show()

    # Инициализация камеры
    cap_operator = cv2.VideoCapture(0)
    
    # Обновление данных
    while True:
        ret, frame = cap_operator.read()
        if not ret:
            break

        # Обновляем изображение в окне видео
        video_window.update_image(frame)
        
        # Обновляем данные на информационном окне
        info_window.update_data(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_operator.release()
    cv2.destroyAllWindows()
    
    sys.exit(app.exec_())
