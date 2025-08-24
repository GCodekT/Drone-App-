import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import clip
from PIL import Image
import os
import numpy as np
import cv2
import time
import asyncio
import concurrent.futures
from torch.utils.data import DataLoader, Dataset
import re
from typing import List, Tuple, Dict, Optional


# Кастомный датасет для Swin-Tiny
class SwinImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.preprocess(image)
            return image, self.image_paths[idx]
        except Exception as e:
            print(f"Ошибка при загрузке {self.image_paths[idx]}: {e}")
            return torch.zeros(3, 224, 224), self.image_paths[idx]


# Кастомный датасет для CLIP
class ClipImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.preprocess(image)
            return image, self.image_paths[idx]
        except Exception as e:
            print(f"Ошибка при загрузке {self.image_paths[idx]}: {e}")
            return torch.zeros(3, 224, 224), self.image_paths[idx]


# Модель Swin-Tiny
class SwinTinyModel(nn.Module):
    def __init__(self):
        super(SwinTinyModel, self).__init__()
        self.backbone = models.swin_t(weights='IMAGENET1K_V1')
        self.backbone.head = nn.Identity()
        self.fc = nn.Linear(768, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


# Косинусное расстояние
def cosine_similarity(features1, features2):
    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    return (features1 @ features2.T).item()


# SIFT для финального уточнения с обработкой ошибок
def sift_similarity(img1_path, img2_path):
    try:
        # Проверяем существование файлов
        if not os.path.exists(img1_path):
            print(f"Файл не найден: {img1_path}")
            return 0.0
        if not os.path.exists(img2_path):
            print(f"Файл не найден: {img2_path}")
            return 0.0

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Проверяем, что изображения загружены корректно
        if img1 is None:
            print(f"Не удалось загрузить изображение: {img1_path}")
            return 0.0
        if img2 is None:
            print(f"Не удалось загрузить изображение: {img2_path}")
            return 0.0

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        return len(matches)
    except Exception as e:
        print(f"Ошибка в SIFT: {e}")
        return 0.0


# Извлечение координат из имени файла тайла
def extract_tile_coordinates(filename: str) -> Optional[Tuple[int, int]]:
    """
    Извлекает координаты (row, col) из имени файла тайла формата имя_row_col.png
    """
    # Ищем паттерн _число_число в конце имени файла
    match = re.search(r'_(\d+)_(\d+)\.(?:png|jpg|jpeg|tiff|tif)$', filename)
    if match:
        try:
            row = int(match.group(1))
            col = int(match.group(2))
            return (row, col)
        except ValueError:
            return None
    return None


# Получение соседних тайлов
def get_neighboring_tiles(base_row: int, base_col: int, max_radius: int = 2) -> List[Tuple[int, int]]:
    """
    Возвращает список координат соседних тайлов вокруг базового тайла
    """
    neighbors = []
    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            if dr == 0 and dc == 0:
                continue  # Пропускаем базовый тайл
            neighbors.append((base_row + dr, base_col + dc))
    return neighbors


# Фильтрация тайлов по доступным файлам
def filter_available_tiles(tile_coords: List[Tuple[int, int]], base_name: str, tiles_folder: str) -> List[str]:
    """
    Фильтрует список координат тайлов, оставляя только существующие файлы
    """
    available_tiles = []
    for row, col in tile_coords:
        tile_filename = f"{base_name}_{row}_{col}.png"
        tile_path = os.path.join(tiles_folder, tile_filename)
        if os.path.exists(tile_path):
            available_tiles.append(tile_filename)
    return available_tiles


# Препроцессинг
swin_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

clip_model_temp, clip_preprocess = clip.load("ViT-B/32", device="cpu")
del clip_model_temp


# Пакетная обработка Swin-Tiny для поиска тайлов
def process_swin_tiny_for_tiles(input_image_path: str, tiles_folder: str, device: str,
                                search_tiles: List[str], batch_size: int = 32) -> Dict[str, float]:
    """
    Обрабатывает входное изображение и ищет его среди указанных тайлов
    """
    print(f"[Swin-Tiny] Начало обработки для поиска тайлов...")

    swin_model = SwinTinyModel().to(device).half()
    swin_model.eval()

    # Проверяем существование входного файла
    if not os.path.exists(input_image_path):
        print(f"Входной файл не найден: {input_image_path}")
        return {}

    # Повороты входного изображения
    rotations = [0, 90, 180, 270]
    swin_input_features_dict = {}

    try:
        input_image_pil = Image.open(input_image_path).convert('RGB')
        for angle in rotations:
            rotated_image = input_image_pil.rotate(angle, expand=True)
            input_image_swin = swin_preprocess(rotated_image).unsqueeze(0).to(device).half()
            with torch.no_grad():
                swin_input_features = swin_model(input_image_swin).float()
            swin_input_features_dict[angle] = swin_input_features
    except Exception as e:
        print(f"Ошибка при обработке входного изображения (Swin-Tiny): {e}")
        return {}

    # Пакетная обработка указанных тайлов
    tile_paths = [os.path.join(tiles_folder, tile_name) for tile_name in search_tiles]

    if not tile_paths:
        print("Не найдено тайлов для поиска")
        return {}

    # Создаем датасет и DataLoader
    dataset = SwinImageDataset(tile_paths, swin_preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Сравнение с тайлами (Swin-Tiny)
    swin_similarities = {}

    try:
        with torch.no_grad():
            for batch_idx, (images_batch, paths_batch) in enumerate(dataloader):
                images_batch = images_batch.to(device).half()
                swin_features_batch = swin_model(images_batch).float()

                # Обрабатываем каждое изображение в батче
                for i, (features, path) in enumerate(zip(swin_features_batch, paths_batch)):
                    filename = os.path.basename(path)
                    # Сравниваем с каждым поворотом входного изображения
                    max_similarity = max(cosine_similarity(swin_input_features_dict[angle], features.unsqueeze(0))
                                         for angle in rotations)
                    swin_similarities[filename] = max_similarity

    except Exception as e:
        print(f"Ошибка при пакетной обработке (Swin-Tiny): {e}")
        return {}

    del swin_model
    torch.cuda.empty_cache()

    print(f"[Swin-Tiny] Обработка завершена")
    return swin_similarities


# Пакетная обработка CLIP для поиска тайлов
def process_clip_for_tiles(input_image_path: str, tiles_folder: str, device: str,
                           search_tiles: List[str], batch_size: int = 32) -> Dict[str, float]:
    """
    Обрабатывает входное изображение и ищет его среди указанных тайлов (CLIP)
    """
    print(f"[CLIP] Начало обработки для поиска тайлов...")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # Проверяем существование входного файла
    if not os.path.exists(input_image_path):
        print(f"Входной файл не найден: {input_image_path}")
        return {}

    # Повороты входного изображения
    rotations = [0, 90, 180, 270]
    clip_input_features_dict = {}

    try:
        input_image_pil = Image.open(input_image_path).convert('RGB')
        for angle in rotations:
            rotated_image = input_image_pil.rotate(angle, expand=True)
            input_image_clip = clip_preprocess(rotated_image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_input_features = clip_model.encode_image(input_image_clip).float()
                clip_input_features /= clip_input_features.norm(dim=-1, keepdim=True)
            clip_input_features_dict[angle] = clip_input_features
    except Exception as e:
        print(f"Ошибка при обработке входного изображения (CLIP): {e}")
        return {}

    # Пакетная обработка указанных тайлов
    tile_paths = [os.path.join(tiles_folder, tile_name) for tile_name in search_tiles]

    if not tile_paths:
        print("Не найдено тайлов для поиска")
        return {}

    # Создаем датасет и DataLoader
    dataset = ClipImageDataset(tile_paths, clip_preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Сравнение с тайлами (CLIP)
    clip_similarities = {}

    try:
        with torch.no_grad():
            for batch_idx, (images_batch, paths_batch) in enumerate(dataloader):
                images_batch = images_batch.to(device)
                clip_features_batch = clip_model.encode_image(images_batch).float()
                clip_features_batch = clip_features_batch / clip_features_batch.norm(dim=-1, keepdim=True)

                # Обрабатываем каждое изображение в батче
                for i, (features, path) in enumerate(zip(clip_features_batch, paths_batch)):
                    filename = os.path.basename(path)
                    features_unsqueezed = features.unsqueeze(0)
                    # Сравниваем с каждым поворотом входного изображения
                    max_similarity = max(cosine_similarity(clip_input_features_dict[angle], features_unsqueezed)
                                         for angle in rotations)
                    clip_similarities[filename] = max_similarity

    except Exception as e:
        print(f"Ошибка при пакетной обработке (CLIP): {e}")
        return {}

    print(f"[CLIP] Обработка завершена")
    return clip_similarities


# Асинхронные обертки
async def run_in_thread(func, *args):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        return await loop.run_in_executor(executor, func, *args)


# ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ФРОНТЕНДА
async def find_drone_photo_locations(drone_photos: List[str], general_photo_path: str,
                                     tiles_folder: str, progress_callback=None) -> List[Tuple[str, int, int]]:
    """
    Находит местоположение фото с дрона на общем фото.

    Args:
        drone_photos (List[str]): Список путей к фото с дрона
        general_photo_path (str): Путь к общему фото
        tiles_folder (str): Папка с нарезанными тайлами
        progress_callback (callable): Функция для обновления прогресса

    Returns:
        List[Tuple[str, int, int]]: Список кортежей (путь_к_фото, row, col) для найденных тайлов
    """

    # Проверка существования файлов и папок
    if not os.path.exists(general_photo_path):
        print(f"Ошибка: Общее изображение не найдено: {general_photo_path}")
        return []

    if not os.path.exists(tiles_folder):
        print(f"Ошибка: Папка с тайлами не найдена: {tiles_folder}")
        return []

    if not os.path.isdir(tiles_folder):
        print(f"Ошибка: Указанный путь не является папкой: {tiles_folder}")
        return []

    # Определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    results = []
    base_name = os.path.splitext(os.path.basename(general_photo_path))[0]

    # Обрабатываем каждое фото с дрона
    total_photos = len(drone_photos)
    for i, drone_photo_path in enumerate(drone_photos):
        if progress_callback:
            progress_callback(f"Обработка фото {i + 1}/{total_photos}", int((i / total_photos) * 100))

        print(f"Обработка фото с дрона: {drone_photo_path}")

        try:
            # Проверяем существование файла дрона
            if not os.path.exists(drone_photo_path):
                print(f"Файл дрона не найден: {drone_photo_path}")
                continue

            # Сначала ищем среди всех тайлов
            all_tile_names = [f for f in os.listdir(tiles_folder)
                              if
                              f.startswith(f"{base_name}_") and f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

            if not all_tile_names:
                print(f"Не найдено тайлов для изображения {base_name}")
                continue

            # Запускаем параллельную обработку Swin-Tiny и CLIP
            swin_task = asyncio.create_task(
                run_in_thread(process_swin_tiny_for_tiles, drone_photo_path, tiles_folder, device, all_tile_names, 32))
            clip_task = asyncio.create_task(
                run_in_thread(process_clip_for_tiles, drone_photo_path, tiles_folder, device, all_tile_names, 32))

            swin_similarities = await swin_task
            clip_similarities = await clip_task

            if not swin_similarities or not clip_similarities:
                print(f"Ошибка при обработке фото {drone_photo_path}")
                continue

            # Объединение результатов
            combined_similarities = {}
            for tile_name in swin_similarities:
                if tile_name in clip_similarities:
                    combined_score = 0.5 * swin_similarities[tile_name] + 0.5 * clip_similarities[tile_name]
                    combined_similarities[tile_name] = combined_score

            # Выбор лучшего кандидата
            if combined_similarities:
                best_tile = max(combined_similarities.items(), key=lambda x: x[1])
                best_tile_name, best_score = best_tile

                print(f"Лучший тайл для {drone_photo_path}: {best_tile_name} (score: {best_score:.4f})")

                # Извлекаем координаты найденного тайла
                tile_coords = extract_tile_coordinates(best_tile_name)
                if tile_coords:
                    row, col = tile_coords

                    # Финальное уточнение с SIFT (обрабатываем ошибки)
                    try:
                        best_tile_path = os.path.join(tiles_folder, best_tile_name)
                        sift_score = sift_similarity(drone_photo_path, best_tile_path)
                        print(f"SIFT совпадений: {sift_score}")
                    except Exception as e:
                        print(f"Ошибка SIFT для {drone_photo_path}: {e}")
                        sift_score = 0

                    # Добавляем результат
                    results.append((drone_photo_path, row, col))

            else:
                print(f"Не найдено совпадений для фото {drone_photo_path}")

        except Exception as e:
            print(f"Ошибка при обработке фото {drone_photo_path}: {e}")
            continue

    if progress_callback:
        progress_callback("Обработка завершена", 100)

    return results


# Синхронная версия для удобства использования
def find_drone_photo_locations_sync(drone_photos: List[str], general_photo_path: str,
                                    tiles_folder: str, progress_callback=None) -> List[Tuple[str, int, int]]:
    """
    Синхронная версия функции для поиска местоположения фото с дрона.

    Args:
        drone_photos (List[str]): Список путей к фото с дрона
        general_photo_path (str): Путь к общему фото
        tiles_folder (str): Папка с нарезанными тайлами
        progress_callback (callable): Функция для обновления прогресса

    Returns:
        List[Tuple[str, int, int]]: Список кортежей (путь_к_фото, row, col) для найденных тайлов
    """
    return asyncio.run(find_drone_photo_locations(drone_photos, general_photo_path, tiles_folder, progress_callback))