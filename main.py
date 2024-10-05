import cv2
import tensorflow as tf
import numpy as np
import json
import base64
from datetime import datetime
import os
import glob
import subprocess
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Путь к папкам с изображениями
fire_dataset_path = "./fire_dataset/fire"
no_fire_dataset_path = "./fire_dataset/no_fire"

# Создание и обучение модели, если модель отсутствует
if not os.path.exists("model.tflite"):
    # Использование простой CNN модели для обнаружения пожара
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Два класса: пожар и нет пожара
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Загрузка данных для обучения
    fire_images = glob.glob(os.path.join(fire_dataset_path, "*.jpg"))
    no_fire_images = glob.glob(os.path.join(no_fire_dataset_path, "*.jpg"))
    images = []
    labels = []

    # Загрузка изображений с пожарами и присвоение меток (1 - пожар)
    for image_path in fire_images:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(1)  # Метка для пожара

    # Загрузка изображений без пожаров и присвоение меток (0 - нет пожара)
    for image_path in no_fire_images:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(0)  # Метка для леса без пожара

    if len(images) == 0:
        raise ValueError("Датасет пустой. Убедитесь, что изображения были загружены корректно.")

    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels).astype('int32')

    # Проверяем форму данных перед обучением
    print(f"Shape of images: {images.shape}")  # Должно быть (количество изображений, 224, 224, 3)
    print(f"Shape of labels: {labels.shape}")  # Должно быть (количество изображений,)

    # Обучение модели и сохранение истории
    history = model.fit(images, labels, epochs=10, batch_size=8, validation_split=0.2)  # Уменьшаем размер батча для уменьшения нагрузки

    # Сохранение модели в формате TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    # Визуализация точности и ошибки обучения
    plt.figure(figsize=(12, 4))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.title('Точность модели')

    # График ошибки
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Ошибка на обучении')
    plt.plot(history.history['val_loss'], label='Ошибка на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.title('Ошибка модели')

    plt.show()

    input("Нажмите Enter, чтобы завершить отображение графиков и перейти к следующему этапу...")
    print("Графики завершены, переходим к следующему этапу.")

# Настройка видеопотока
cap = cv2.VideoCapture(0)  # Замените "0" на URL вашей камеры, если нужно

# Загрузка модели (замените путь на вашу модель)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Функция предобработки кадра для подачи в модель
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Замените размер в соответствии с требованиями вашей модели
    input_data = np.expand_dims(resized_frame, axis=0)
    input_data = input_data.astype(np.float32) / 255.0  # Нормализация кадра, если необходимо
    return input_data

# Функция анализа выходных данных модели
def analyze_output(output_data, frame):
    # Пример анализа выходных данных - замените на вашу логику
    # Предположим, что модель возвращает вероятность наличия пожара и отсутствия пожара
    no_fire_probability = output_data[0][0]
    fire_probability = output_data[0][1]

    if fire_probability > 0.8:  # Повысили порог для уменьшения ложных срабатываний
        print(f"Пожар обнаружен с вероятностью {fire_probability:.2f}")
    elif no_fire_probability > 0.8:
        print(f"Нет пожара с вероятностью {no_fire_probability:.2f}")

# Главный цикл получения видеопотока и обработки кадров
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка кадра
    input_data = preprocess_frame(frame)

    # Передача кадра в модель
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Получение результата
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Анализ результата
    analyze_output(output_data, frame)

    # Отображение видеопотока
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Тестирование модели на датасете с пожарами
def test_with_fire_dataset():
    all_images = fire_images + no_fire_images
    for image_path in all_images:
        image = cv2.imread(image_path)
        if image is not None:
            input_data = preprocess_frame(image)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            analyze_output(output_data, image)

test_with_fire_dataset()