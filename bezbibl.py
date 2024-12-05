import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция для показа изображения
def show_image(image, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Загрузка изображения
image_path = "anfas.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Функция интегрального изображения
def compute_integral_image(image: np.ndarray) -> np.ndarray:
    return cv2.integral(image)

# Haar-признаки
def haar_features(integral_image, x, y, w, h):
    mid_h = h // 2
    white = integral_image[y + mid_h, x + w] - integral_image[y + mid_h, x] \
            - integral_image[y, x + w] + integral_image[y, x]
    black = integral_image[y + h, x + w] - integral_image[y + h, x] \
            - integral_image[y + mid_h, x + w] + integral_image[y + mid_h, x]
    return black - white

# Фильтрация пересекающихся окон (Non-Maximum Suppression, NMS)
def non_maximum_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(areas)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]
        order = order[np.where(overlap <= overlap_thresh)[0] + 1]

    return boxes[keep]

# Детекция лиц с фильтрацией
def detect_faces(image, window_size, threshold):
    rows, cols = image.shape
    win_w, win_h = window_size
    integral_image = compute_integral_image(image)
    detections = []

    step_size = 10  # Увеличенный шаг окна сканирования
    for y in range(0, rows - win_h, step_size):
        for x in range(0, cols - win_w, step_size):
            feature = haar_features(integral_image, x, y, win_w, win_h)
            if feature > threshold:
                detections.append((x, y, win_w, win_h))

    # Применяем NMS для уменьшения количества пересекающихся окон
    detections = non_maximum_suppression(detections)
    return detections



# Детекция лиц
window_size = (600, 600)
threshold = 5000  # Увеличенный порог
detected_faces = detect_faces(image, window_size, threshold)

# Отображение результата
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in detected_faces:
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Показ изображения
show_image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), title="without library")
