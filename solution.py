import numpy as np
import pandas as pd
from typing import List, Union
import cv2
import torch
import os
import csv
import torch
from ultralytics import YOLO

device = 'cpu'

if torch.cuda.is_available():
    print("Загрузка модели на CUDA")
    device = 'cuda'
else: print("CUDA не обнаружена, загрузка модели на CPU")

model = YOLO("best.pt").to(device)


def slice_image(image, patch_size=640, overlap=0.2):
    """
    Нарезка входного тестового изображения на патчи фиксированного размера с перекрытием.

    :param image: Изображение в формате numpy.ndarray.
    :param patch_size: Размер патча (по умолчанию 640x640).
    :param overlap: Процент перекрытия между патчами (по умолчанию 20%).
    :return: Список патчей изображения.
    """
    if image is None:
        raise ValueError("Не удалось загрузить изображение.")

    h, w, _ = image.shape
    step_size = int(patch_size * (1 - overlap))

    patches = []
    coordinates = []

    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)

            patch = image[y:y_end, x:x_end]
            patch_height, patch_width, _ = patch.shape
            
            if patch_height < patch_size or patch_width < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch_height, :patch_width] = patch
                patch = padded_patch

            patches.append(patch)
            coordinates.append((x, y, x_end, y_end))

    return patches, coordinates, image


def process_yolo_patches(patches, model):
    """
    Обработка патчей с использованием YOLOv10.

    :param patches: Список патчей изображений.
    :return: Результаты предсказаний для каждого патча.
    """
    results = []

    # Прогоняем каждый патч через модель YOLO
    for patch in patches:
        result = model(patch)  # Предсказания для одного патча
        results.append(result[0])

    return results


def non_maximum_suppression(predictions, iou_threshold=0.5):
    """
    Применение Non-Maximum Suppression для удаления дублирующих рамок.

    :param predictions: Тензор с предсказаниями (x1, y1, x2, y2, conf, class_id) для всех патчей.
    :param iou_threshold: Порог IoU для NMS (по умолчанию 0.5).
    :return: Тензор с рамками после применения NMS.
    """
    if len(predictions) == 0:
        return predictions

    # Применяем встроенный метод PyTorch для NMS
    boxes = predictions[:, :4]  # Координаты боксов
    scores = predictions[:, 4]  # Уверенность
    indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)  # Применение NMS

    return predictions[indices]


def postprocess_patches(original_image, predictions, coordinates, threshold=0.6, iou_threshold=0.5):
    all_predictions = []
    results = []

    for i, result in enumerate(predictions):
        x_offset, y_offset, _, _ = coordinates[i]

        for detection in result.boxes.data:  # Получаем детекции для патча
            x1, y1, x2, y2, conf, class_id = detection[:6].tolist()  # Извлекаем данные

            if conf >= threshold:
                # Преобразуем координаты обратно в оригинальное изображение
                x1, y1, x2, y2 = int(x1 + x_offset), int(y1 + y_offset), int(x2 + x_offset), int(y2 + y_offset)

                # Добавляем предсказание (x1, y1, x2, y2, conf, class_id) в общий список
                all_predictions.append([x1, y1, x2, y2, conf, class_id])

    # Преобразуем в тензор для применения NMS
    all_predictions = torch.tensor(all_predictions)

    # Применяем NMS
    nms_predictions = non_maximum_suppression(all_predictions, iou_threshold)

    for detection in nms_predictions:
        x1, y1, x2, y2, conf, class_id = map(float, detection[:6])

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        result_dict = {
            'xc': round(xc / original_image.shape[1], 4),
            'yc': round(yc / original_image.shape[0], 4),
            'w': round(w / original_image.shape[1], 4),
            'h': round(h / original_image.shape[0], 4),
            'label': int(class_id),
            'score': round(conf, 4)
        }
        results.append(result_dict)
    return results


def predict(images: Union[List[np.ndarray], np.ndarray]) -> dict:
    """
    Обработка изображений с помощью YOLO и возвращение предсказаний.

    :param images: Список изображений в формате numpy.ndarray или одно изображение.
    :return: Словарь с предсказаниями для каждого изображения.
    """
    results_data = []

    # Если images не является списком, преобразуем его в список
    if isinstance(images, np.ndarray):
        images = [images]

    for idx, image in enumerate(images):
        # Нарезаем изображение на патчи
        patches, coordinates, original_image = slice_image(image)
        # Обрабатываем патчи через YOLO
        yolo_results = process_yolo_patches(patches, model)
        # Постобработка
        result_dict = postprocess_patches(original_image, yolo_results, coordinates)

        results_data.extend(result_dict)

    return results_data


def create_solution_submission(images: Union[List[np.ndarray], np.ndarray], output_csv='solution_submission.csv'):
    """
    Обработка изображений и сохранение результатов в CSV.

    :param images: Список изображений в формате numpy.ndarray или одно изображение.
    :param output_csv: Имя выходного CSV-файла.
    """
    # Получаем предсказания
    results_data = predict(images)

    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)

def run_test_solution():
    folder_path = 'images'
    output_csv = 'solution_submission.csv'

    images = []
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

    # Вызов функции для предсказания и сохранения в CSV
    create_solution_submission(images, output_csv)
    
    with open("solution_submission.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print(row)

if __name__ == '__main__':
    run_test_solution()