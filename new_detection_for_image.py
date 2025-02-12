import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
import torch
import os

# Загружаем модель YOLOv11
model = YOLO("best_yolo11.pt")


def slice_image(image_path, patch_size=640, overlap=0.2):
    """
    Нарезка входного тестового изображения на патчи фиксированного размера с перекрытием.

    :param image_path: Путь к тестовому изображению.
    :param patch_size: Размер патча (по умолчанию 640x640).
    :param overlap: Процент перекрытия между патчами (по умолчанию 20%).
    :return: Список патчей изображения.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

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
            coordinates.append((x, y, x_end, y_end))  # сохраняем координаты патча

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


def non_maximum_suppression(predictions, iou_threshold=0.1):
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


def postprocess_patches(image_file_name, original_image, predictions, coordinates, threshold=0.6, iou_threshold=0.2):
    all_predictions = []
    results = []
    image_id = os.path.splitext(os.path.basename(image_file_name))[0]

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
            'image_id': image_id,
            'xc': round(xc / original_image.shape[1], 4),
            'yc': round(yc / original_image.shape[0], 4),
            'w': round(w / original_image.shape[1], 4),
            'h': round(h / original_image.shape[0], 4),
            'label': int(class_id),
            'score': round(conf, 4)
        }
        results.append(result_dict)
    return results


def predict(folder_path):
    results_data = []

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Нарезаем изображение на патчи
        patches, coordinates, original_image = slice_image(image_path)
        # Обрабатываем патчи через YOLO
        yolo_results = process_yolo_patches(patches, model)
        # постобработка
        result_dict = postprocess_patches(image_file, original_image, yolo_results, coordinates)

        results_data.extend(result_dict)

    return results_data


def create_solution_submission(folder_path, output_csv=None, output_images_folder='output_images2'):
    results_data = []

    # Создаем папку для сохранения изображений, если её нет
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Нарезаем изображение на патчи
        patches, coordinates, original_image = slice_image(image_path)
        # Обрабатываем патчи через YOLO
        yolo_results = process_yolo_patches(patches, model)
        # Постобработка
        result_dict = postprocess_patches(image_file, original_image, yolo_results, coordinates)

        results_data.extend(result_dict)

        # Визуализируем bbox на изображении
        for detection in result_dict:
            x1 = int(detection['xc'] * original_image.shape[1] - (detection['w'] * original_image.shape[1]) / 2)
            y1 = int(detection['yc'] * original_image.shape[0] - (detection['h'] * original_image.shape[0]) / 2)
            x2 = int(x1 + detection['w'] * original_image.shape[1])
            y2 = int(y1 + detection['h'] * original_image.shape[0])
            label = detection['label']
            score = detection['score']

            # Рисуем bbox
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{label}: {round(score, 2)}"
            cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Сохраняем визуализированное изображение
        output_image_path = os.path.join(output_images_folder, image_file)
        cv2.imwrite(output_image_path, original_image)

    if not output_csv is None:
        # Сохраняем результаты в CSV
        df = pd.DataFrame(results_data)
        df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    create_solution_submission(r'C:\Users\kitar\Downloads\000000.jpg')
