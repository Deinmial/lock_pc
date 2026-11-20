#!/usr/bin/env python3
import cv2
import time
import shutil
import platform
import subprocess
import os
import numpy as np

CHECK_INTERVAL = 5 # Проверяем каждые 5 секунд
ABSENT_LIMIT = 30 # Через сколько секунд блокируем
CAMERA_INDEX = 0 # Индекс вебкамеры (0 - по умолчанию)
BG_CHANGE_RATIO = 0.20 # Порог изменения фона (20%)
MOTION_RATIO = 0.04 # Порог движения (4%)
YUNET_SCORE = 0.5 # Порог уверенности нейросети (50%)
DEBUG = False

# Пути
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL = os.path.join(SCRIPT_DIR, 'face_detection_yunet_2023mar.onnx')
BG_FILE = os.path.join(SCRIPT_DIR, 'background.png')

IS_WINDOWS = platform.system() == 'Windows'

# Каскады Хаара как запасной аэродром (При наклоне головы он умнее...)
upperbody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml'
)

# Блокировка экрана
def lock_screen():
    # Windows
    if IS_WINDOWS:
        try:
            subprocess.run(['rundll32.exe', 'user32.dll,LockWorkStation'], check=True, timeout=10)
            print('[lock] Windows: экран заблокирован')
            return True
        except Exception as e:
            print(f'[lock] Windows: не удалось заблокировать: {e}')
            return False

    # Linux
    candidates = []
    if shutil.which('loginctl'):
        candidates.append(['loginctl', 'lock-session'])
    if shutil.which('gnome-screensaver-command'):
        candidates.append(['gnome-screensaver-command', '-l'])
    if shutil.which('xdg-screensaver'):
        candidates.append(['xdg-screensaver', 'lock'])
    if shutil.which('xflock4'):
        candidates.append(['xflock4'])
    if shutil.which('cinnamon-screensaver-command'):
        candidates.append(['cinnamon-screensaver-command', '-l'])
    if shutil.which('mate-screensaver-command'):
        candidates.append(['mate-screensaver-command', '-l'])
    if shutil.which('swaylock'):
        candidates.append(['swaylock'])

    for cmd in candidates:
        try:
            subprocess.run(cmd, check=True, timeout=10)
            print(f'[lock] Заблокировано через: {" ".join(cmd)}')
            return True
        except Exception as e:
            print(f'[lock] {cmd} не сработало: {e}')
    print('[lock] Не удалось заблокировать экран')
    return False


# Детекторы
def make_detector():
    if not hasattr(cv2, 'FaceDetectorYN'):
        raise RuntimeError('cv2.FaceDetectorYN недоступен — нужен OpenCV >= 4.5.4')

    if not os.path.exists(YUNET_MODEL):
        raise RuntimeError(
            f'Не найдена модель YuNet: {YUNET_MODEL}\n'
            f'Скачать: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
        )

    return cv2.FaceDetectorYN.create(
        YUNET_MODEL, '', (320, 320), # Сжимаем кадр до 320x320
        score_threshold=YUNET_SCORE, # Порог уверенности
        nms_threshold=0.3,           # NMS
        top_k=5000,                  # Макс количество лиц, которе вернётся за кадр (5000 - с запасом)
    )

# Фильтр камеры
def grab_gray(cap):
    for _ in range(5): # Прогрев камеры
        cap.read()
    ret, frame = cap.read() # Чтение кадра
    if not ret:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Перевод в grayscale
    gray = cv2.equalizeHist(gray)
    return frame, gray

# Поиск лица (yunet)
def detect_face_yunet(detector, frame):
    h, w = frame.shape[:2] # Высота, ширина кадра
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame) # Скан изображения
    if faces is None:
        return False, []
    return len(faces) > 0, faces

# Поиск лица (Хаара)
def presence_by_upperbody(gray):
    h, w = gray.shape
    small = cv2.resize(gray, (w // 2, h // 2)) # Уменьшаем изображение (для оптимизации)
    bodies = upperbody_cascade.detectMultiScale(small, scaleFactor=1.05, minNeighbors=4, minSize=(40, 20)) # Распознавание
    if DEBUG:
        print(f'[upperbody] найдено: {len(bodies)}')
    return len(bodies) > 0

# Детекция движения
def presence_by_motion(prev_gray, gray):
    if prev_gray is None: # Первый запуск (т.к. не с чем сравнивать)
        return False
    prev_b = cv2.GaussianBlur(prev_gray, (21, 21), 0) # Убираем
    gray_b = cv2.GaussianBlur(gray, (21, 21), 0)      # шум
    diff = cv2.absdiff(prev_b, gray_b) # Ищем отличия
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # Бинаризация
    ratio = np.count_nonzero(thresh) / gray.size # Процент измениний
    if DEBUG:
        print(f'[motion] ratio={ratio:.4f} (порог {MOTION_RATIO})')
    return ratio > MOTION_RATIO

# Проверка статичного присутствия
def presence_by_background(bg, gray):
    if bg is None: # Проверка наличия эталона
        return False
    diff = cv2.absdiff(bg, gray) # Сравниваем с пустой комнатой
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # Фильтрация шума
    ratio = np.count_nonzero(thresh) / gray.size # Площадь захвата
    if DEBUG:
        print(f'[bg-diff] ratio={ratio:.4f} (порог {BG_CHANGE_RATIO})')
    return ratio > BG_CHANGE_RATIO

# Калибровка
def calibrate_background(cap):
    print('[calib] Отойдите от компьютера на 5 секунд — происходит фиксирование эталонного пустого фона...')
    samples = []
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.equalizeHist(g)
            samples.append(g.astype(np.float32)) # Делаем 10 кадров и перводим их в Ч/Б с контрастом
        time.sleep(0.5)
    if not samples:
        return None
    bg = np.mean(samples, axis=0).astype(np.uint8) # Среднее арифмитическое между 10 кадрами
    cv2.imwrite(BG_FILE, bg) # Сохраняем эталон
    print(f'[calib] Эталон сохранён в {BG_FILE}')
    return bg

# Проверяем
def person_present(detector, cap, bg, prev_gray):
    frame, gray = grab_gray(cap) # Получаем кадр
    if gray is None:
        return True, 'camera-error', prev_gray, frame

    #YuNet
    found_face, _ = detect_face_yunet(detector, frame)
    if found_face:
        return True, 'yunet-face', gray, frame

    # Каскады Хаара
    if presence_by_upperbody(gray):
        return True, 'upperbody', gray, frame

    # Детектор движения
    if presence_by_motion(prev_gray, gray):
        return True, 'motion', gray, frame

    # Сравнение с пустым фоном
    if presence_by_background(bg, gray):
        return True, 'bg-diff', gray, frame

    # Человек не обнаружен
    return False, 'none', gray, frame


print(f'[main] ОС: {platform.system()}')
print(f'[main] Старт: проверка каждые {CHECK_INTERVAL}с, блокировка после {ABSENT_LIMIT}с без человека')

detector = make_detector()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print('[main] Камера недоступна — выходим')
    raise SystemExit(1)

try:
    input('[main] Enter — снять эталон пустого фона (Ctrl+C — пропустить)... ')
    bg = calibrate_background(cap)
    print('[main] Программа запущена')
except KeyboardInterrupt:
    print('\n[main] Калибровка пропущена')
    print('[main] Программа запущена')
    bg = None # Продолжаем без калибровки

prev_gray = None # Предыдущий кадр
absent_since = None # Текущий кадр
is_locked = False  # Флаг состояния блокировки

try:
    while True:
        # Пока не нажат Ctrl+C
        present, reason, gray, frame = person_present(detector, cap, bg, prev_gray)
        now = time.time()

        if present:
            if is_locked or absent_since is not None:
                print(f'[main] Человек снова в кадре ({reason})')
            is_locked = False
            absent_since = None
        else:
            if absent_since is None:
                absent_since = now
                print('[main] Человека не видно, начинаю отсчёт')
            elif not is_locked:
                elapsed = now - absent_since
                print(f'[main] Нет человека {elapsed:.0f}с / {ABSENT_LIMIT}с')
                if elapsed >= ABSENT_LIMIT:
                    lock_screen()
                    is_locked = True
                    print('[main] Экран заблокирован. Ожидание возвращения пользователя...')
                    # absent_since не сбрасываем, чтобы не начинать отсчет заново
                    # is_locked защищает от повторного вызова lock_screen()

        if DEBUG and frame is not None:
            _, faces = detect_face_yunet(detector, frame)
            if faces is not None:
                for f in faces:
                    x, y, w, h = f[:4].astype(int)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, reason, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('lock.py', frame)
            cv2.waitKey(1)
        prev_gray = gray
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print('\n[main] Программа остановлена пользователем')
finally:
    cap.release() # Осовобождаем веб камеру
    if DEBUG:
        cv2.destroyAllWindows()
