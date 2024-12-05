import cv2
import matplotlib.pyplot as plt

# Загрузка классификатора Виолы-Джонса для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения
image_path = "anfas.jpg"  # Укажите путь к изображению
image = cv2.imread(image_path)

# Преобразование изображения в градации серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Детекция лиц
faces = face_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(500, 500)
)

# Рисование прямоугольников вокруг обнаруженных лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Преобразование BGR в RGB для отображения с Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Функция для показа изображения
def show_image(image, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Отображение изображения с выделенными лицами
show_image(image_rgb, title="Using library")
