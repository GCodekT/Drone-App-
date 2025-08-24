import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import shutil
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading


class ProgressWindow(tk.Toplevel):
    def __init__(self, parent, title="Обработка"):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x130")
        self.resizable(False, False)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=20)

        self.label = ttk.Label(self, text="Подготовка к обработке...")
        self.label.pack()

        self.grab_set()  # Блокирует главное окно
        self.cancelled = False
        ttk.Button(self, text="Отмена", command=self.cancel).pack(pady=5)

    def cancel(self):
        self.cancelled = True
        self.destroy()


class ProcessingFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Стилизация в стиле Material Design
        self.configure(style='Card.TFrame')

        self.status_label = ttk.Label(self, text="Идет обработка изображений...",
                                      font=('Segoe UI', 12, 'bold'))
        self.status_label.pack(pady=20)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        self.progress["maximum"] = 100

        self.detail_label = ttk.Label(self, text="Пожалуйста, подождите...")
        self.detail_label.pack()

    def update_progress(self, message, percent):
        """Обновление прогресса"""
        self.status_label.config(text=message)
        self.progress["value"] = percent
        self.detail_label.config(text=f"Прогресс: {percent}%")
        self.update()


class ZoomableMarkedImageCanvas(tk.Canvas):
    def __init__(self, parent, image_path, found_tiles, photo_numbers, tile_size=256):
        super().__init__(parent, bg='#f8f9fa', highlightthickness=0)
        self.image_path = image_path
        self.found_tiles = found_tiles
        self.photo_numbers = photo_numbers
        self.tile_size = tile_size

        # Параметры зума и перемещения
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        self.dragging = False

        # Кэширование изображений
        self.original_image = None
        self.display_image = None
        self.photo = None

        # Таймер для отложенной отрисовки
        self.redraw_timer = None

        # Кэширование масштабированных изображений
        self.cached_scaled_images = {}  # Кэш для разных масштабов
        self.current_cache_key = None

        self.load_image()
        self.setup_bindings()
        self.draw_image()

    def load_image(self):
        """Загрузка изображения"""
        try:
            self.original_image = Image.open(self.image_path)
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")

    def setup_bindings(self):
        """Настройка обработчиков событий"""
        # Зум колесиком мыши
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-4>", self.on_mousewheel)
        self.bind("<Button-5>", self.on_mousewheel)

        # Перемещение перетаскиванием
        self.bind("<Button-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_button_motion)
        self.bind("<ButtonRelease-1>", self.on_button_release)

        # Изменение размера
        self.bind("<Configure>", self.on_resize)

    def on_mousewheel(self, event):
        """Обработка колесика мыши для зума"""
        if not self.original_image:
            return

        # Сохраняем текущие координаты курсора
        canvas_x = event.x
        canvas_y = event.y

        # Определяем направление прокрутки
        if event.num == 5 or event.delta < 0:
            scale_factor = 0.9
        elif event.num == 4 or event.delta > 0:
            scale_factor = 1.1
        else:
            return

        # Вычисляем новые координаты после масштабирования
        old_scale = self.scale
        self.scale *= scale_factor

        # Ограничиваем зум
        self.scale = max(self.min_scale, min(self.scale, self.max_scale))

        # Корректируем смещение для зума относительно курсора
        self.offset_x = canvas_x - (canvas_x - self.offset_x) * (self.scale / old_scale)
        self.offset_y = canvas_y - (canvas_y - self.offset_y) * (self.scale / old_scale)

        # Ограничиваем перемещение
        self.constrain_offset()

        # Отложенная отрисовка с высокой частотой
        self.schedule_redraw(high_priority=True)

    def on_button_press(self, event):
        """Начало перетаскивания"""
        self.dragging = True
        self.last_x = event.x
        self.last_y = event.y
        self.configure(cursor="fleur")

    def on_button_motion(self, event):
        """Перетаскивание изображения"""
        if self.dragging and self.original_image:
            dx = event.x - self.last_x
            dy = event.y - self.last_y

            self.offset_x += dx
            self.offset_y += dy

            self.last_x = event.x
            self.last_y = event.y

            # Ограничиваем перемещение
            self.constrain_offset()

            # Высокочастотная отрисовка при перетаскивании
            self.schedule_redraw(high_priority=True)

    def on_button_release(self, event):
        """Окончание перетаскивания"""
        self.dragging = False
        self.configure(cursor="")

    def on_resize(self, event):
        """Обработка изменения размера холста"""
        # Отложенная отрисовка
        self.schedule_redraw()

    def constrain_offset(self):
        """Ограничивает перемещение чтобы не выходить за границы изображения"""
        if not self.original_image:
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        scaled_width = self.original_image.width * self.scale
        scaled_height = self.original_image.height * self.scale

        # Ограничиваем смещение чтобы не уходить слишком далеко за границы
        max_offset_x = canvas_width * 0.5
        min_offset_x = -(scaled_width - canvas_width * 0.5)
        max_offset_y = canvas_height * 0.5
        min_offset_y = -(scaled_height - canvas_height * 0.5)

        self.offset_x = max(min_offset_x, min(max_offset_x, self.offset_x))
        self.offset_y = max(min_offset_y, min(max_offset_y, self.offset_y))

    def schedule_redraw(self, high_priority=False):
        """Отложенная отрисовка с высокой частотой"""
        if self.redraw_timer:
            self.after_cancel(self.redraw_timer)

        # Высокая частота обновления (до 400 FPS)
        delay = 2 if high_priority else 16  # 2мс для высокого приоритета, 16мс для обычного (~60 FPS)
        self.redraw_timer = self.after(delay, self.draw_image)

    def get_cached_scaled_image(self, target_width, target_height):
        """Получение кэшированного масштабированного изображения"""
        cache_key = f"{target_width}x{target_height}"

        # Если есть кэш и он подходит, используем его
        if cache_key in self.cached_scaled_images:
            return self.cached_scaled_images[cache_key]

        # Создаем новое масштабированное изображение
        if target_width > 2000 or target_height > 2000:
            scaled_image = self.original_image.resize((target_width, target_height), Image.BILINEAR)
        else:
            scaled_image = self.original_image.resize((target_width, target_height), Image.LANCZOS)

        # Сохраняем в кэш (ограничиваем размер кэша)
        if len(self.cached_scaled_images) > 3:  # Ограничиваем кэш 3 изображениями
            # Удаляем самый старый элемент
            oldest_key = next(iter(self.cached_scaled_images))
            del self.cached_scaled_images[oldest_key]

        self.cached_scaled_images[cache_key] = scaled_image
        self.current_cache_key = cache_key

        return scaled_image

    def draw_image(self):
        """Отрисовка изображения с учетом зума и смещения"""
        # Отменяем запланированную отрисовку
        if self.redraw_timer:
            self.after_cancel(self.redraw_timer)
            self.redraw_timer = None

        if not self.original_image:
            return

        # Получаем размеры холста
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Масштабируем изображение
        target_width = int(self.original_image.width * self.scale)
        target_height = int(self.original_image.height * self.scale)

        # Избегаем слишком маленьких размеров
        target_width = max(1, target_width)
        target_height = max(1, target_height)

        # Получаем кэшированное масштабированное изображение
        try:
            scaled_image = self.get_cached_scaled_image(target_width, target_height)
        except:
            # Если кэширование не удалось, создаем напрямую
            if target_width > 2000 or target_height > 2000:
                scaled_image = self.original_image.resize((target_width, target_height), Image.BILINEAR)
            else:
                scaled_image = self.original_image.resize((target_width, target_height), Image.LANCZOS)

        # Создаем копию для рисования маркеров (только видимые маркеры)
        marked_image = scaled_image.copy()
        draw = ImageDraw.Draw(marked_image)

        # Рисуем только видимые маркеры для оптимизации
        self.draw_visible_markers(draw, target_width, target_height)

        # Конвертируем в PhotoImage
        try:
            self.photo = ImageTk.PhotoImage(marked_image)
        except:
            # Если изображение слишком большое, создаем уменьшенную версию
            if target_width > 4000 or target_height > 4000:
                small_image = marked_image.resize((target_width // 2, target_height // 2), Image.BILINEAR)
                self.photo = ImageTk.PhotoImage(small_image)
            else:
                return

        # Очищаем холст и рисуем изображение
        self.delete("all")

        # Рисуем изображение с учетом смещения
        self.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW)

        # Добавляем инструкции при нормальном масштабе
        if abs(self.scale - 1.0) < 0.1 and abs(self.offset_x) < 100 and abs(self.offset_y) < 100:
            self.create_text(
                canvas_width // 2, canvas_height - 30,
                text="Колесо мыши: масштаб | Перетащить: перемещение",
                fill="#666666",
                font=('Segoe UI', 9)
            )

    def draw_visible_markers(self, draw, scaled_width, scaled_height):
        """Рисует только видимые маркеры для оптимизации производительности"""
        if not self.original_image:
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Вычисляем видимую область
        visible_left = max(0, -self.offset_x)
        visible_top = max(0, -self.offset_y)
        visible_right = min(scaled_width, canvas_width - self.offset_x)
        visible_bottom = min(scaled_height, canvas_height - self.offset_y)

        # Рисуем маркеры только для видимой области
        for row, col in self.found_tiles:
            # Преобразуем координаты тайлов в координаты масштабированного изображения
            x = int((col * self.tile_size + self.tile_size // 2) * self.scale)
            y = int((row * self.tile_size + self.tile_size // 2) * self.scale)

            # Проверяем, находится ли маркер в видимой области (с запасом)
            margin = 50  # Запас для маркеров чуть за пределами видимости
            if (visible_left - margin <= x <= visible_right + margin and
                    visible_top - margin <= y <= visible_bottom + margin):

                # Рисуем фиолетовый круг
                radius = max(3, int(25 * self.scale))  # Минимальный размер для видимости
                if radius > 1:  # Не рисуем слишком маленькие маркеры
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                                 fill="#9c27b0", outline="white", width=max(1, int(2 * self.scale)))

                    # Получаем номер фото
                    photo_number = self.photo_numbers.get((row, col), "?")

                    # Рисуем белый текст с номером (только для достаточно больших маркеров)
                    if radius > 8:
                        try:
                            # Адаптивный размер шрифта
                            font_size = max(6, int(20 * self.scale))
                            if font_size >= 8:
                                font = ImageFont.truetype("arial.ttf", font_size)
                            else:
                                font = ImageFont.load_default()
                        except:
                            try:
                                font = ImageFont.truetype("DejaVuSans.ttf", max(6, int(20 * self.scale)))
                            except:
                                font = ImageFont.load_default()

                        # Центрируем текст
                        try:
                            bbox = draw.textbbox((0, 0), str(photo_number), font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        except:
                            text_width, text_height = draw.textsize(str(photo_number), font=font)

                        text_x = x - text_width // 2
                        text_y = y - text_height // 2

                        draw.text((text_x, text_y), str(photo_number), fill="white", font=font)


class DronePhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Photo App")
        self.root.geometry("900x700")

        # Настройка цветовой палитры
        self.setup_colors()
        self.setup_styles()

        # Переменные для хранения путей к изображениям
        self.drone_photos = []
        self.general_photo = None
        self.tiles_folder = "base"  # Папка для тайлов
        self.temp_folder = "temp"  # Папка для временных файлов
        self.found_coordinates = {}  # Словарь для хранения найденных координат
        self.found_tiles = []  # Список найденных тайлов (row, col)
        self.photo_numbers = {}  # Словарь для хранения номеров фото

        self.create_first_window()

    def setup_colors(self):
        """Настройка цветовой палитры"""
        self.colors = {
            'primary': '#4caf50',  # Зеленый (основной)
            'secondary': '#2196f3',  # Голубой (вторичный)
            'accent': '#9c27b0',  # Фиолетовый (акцент)
            'background': '#f5f5f5',  # Фон
            'surface': '#ffffff',  # Поверхность
            'text_primary': '#212121',  # Основной текст
            'text_secondary': '#757575'  # Вторичный текст
        }

    def setup_styles(self):
        """Настройка стилей в стиле Material Design"""
        style = ttk.Style()

        # Основные стили
        style.configure('TFrame', background=self.colors['background'])
        style.configure('Card.TFrame', background=self.colors['surface'], relief='flat')
        style.configure('Header.TLabel',
                        font=('Segoe UI', 16, 'bold'),
                        background=self.colors['surface'],
                        foreground=self.colors['text_primary'])
        style.configure('Title.TLabel',
                        font=('Segoe UI', 14, 'bold'),
                        background=self.colors['surface'],
                        foreground=self.colors['primary'])
        style.configure('TLabel',
                        font=('Segoe UI', 10),
                        background=self.colors['background'],
                        foreground=self.colors['text_primary'])

        # Стили кнопок
        style.configure('TButton',
                        font=('Segoe UI', 10),
                        padding=6,
                        background=self.colors['surface'])
        style.configure('Primary.TButton',
                        font=('Segoe UI', 10, 'bold'),
                        padding=6,
                        background=self.colors['primary'],
                        foreground='white')
        style.map('Primary.TButton',
                  background=[('active', '#43a047')])

        # Стили для LabelFrame
        style.configure('TLabelframe',
                        background=self.colors['background'],
                        relief='flat')
        style.configure('TLabelframe.Label',
                        font=('Segoe UI', 11, 'bold'),
                        background=self.colors['background'],
                        foreground=self.colors['primary'])

    def create_first_window(self):
        """Создаем первое окно для загрузки фото"""
        self.clear_window()
        self.root.geometry("900x700")

        # Основной контейнер
        main_container = ttk.Frame(self.root, style='Card.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Заголовок
        header = ttk.Label(main_container, text="Drone Photo Analyzer", style='Header.TLabel')
        header.pack(pady=(20, 30))

        # Контент
        content_frame = ttk.Frame(main_container, style='Card.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)

        # Фрейм для загрузки фото дрона
        drone_frame = ttk.LabelFrame(content_frame, text="Фото с дрона", padding=20)
        drone_frame.pack(fill=tk.X, pady=(0, 20))

        # Кнопки для загрузки фото с дрона
        button_frame = ttk.Frame(drone_frame)
        button_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(button_frame, text="Выбрать файлы",
                   style='Primary.TButton',
                   command=self.load_drone_photos).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Выбрать папку",
                   style='TButton',
                   command=self.load_drone_folder).pack(side=tk.LEFT)

        # Лейбл для отображения количества загруженных фото
        self.drone_count_label = ttk.Label(drone_frame, text="Фото не загружены")
        self.drone_count_label.pack(anchor=tk.E)

        # Фрейм для общего фото
        general_frame = ttk.LabelFrame(content_frame, text="Общее фото", padding=20)
        general_frame.pack(fill=tk.X, pady=(0, 30))

        ttk.Button(general_frame, text="Выбрать файл",
                   style='Primary.TButton',
                   command=self.load_general_photo).pack(side=tk.LEFT)

        # Кнопка далее
        self.next_btn = ttk.Button(content_frame, text="Далее",
                                   style='Primary.TButton',
                                   command=self.start_processing,
                                   state=tk.DISABLED)
        self.next_btn.pack(side=tk.BOTTOM, anchor=tk.SE)

    def load_drone_photos(self):
        """Загрузка нескольких фото с дрона"""
        files = filedialog.askopenfilenames(
            title="Выберите фото с дрона",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.tiff *.tif"), ("All files", "*.*"))
        )
        if files:
            # Копируем файлы во временную папку
            self.drone_photos = self.copy_files_to_temp(files)
            self.update_drone_count()
            self.check_next_button()

    def load_drone_folder(self):
        """Загрузка всех фото из выбранной папки"""
        folder = filedialog.askdirectory(title="Выберите папку с фото дрона")
        if folder:
            files = []
            for file in os.listdir(folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                    files.append(os.path.join(folder, file))

            if not files:
                messagebox.showwarning("Внимание", "В папке не найдены изображения!")
            else:
                # Копируем файлы во временную папку
                self.drone_photos = self.copy_files_to_temp(files)
                self.update_drone_count()
                self.check_next_button()

    def copy_files_to_temp(self, files):
        """Копирование файлов во временную папку"""
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

        copied_files = []
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.temp_folder, filename)

                # Если файл уже существует, добавляем суффикс
                counter = 1
                original_dest_path = dest_path
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(original_dest_path)
                    dest_path = f"{name}_{counter}{ext}"
                    counter += 1

                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
            except Exception as e:
                print(f"Ошибка копирования файла {file_path}: {e}")

        return copied_files

    def update_drone_count(self):
        """Обновляет лейбл с количеством загруженных фото"""
        count = len(self.drone_photos)
        self.drone_count_label.config(text=f"Загружено фото: {count}")

    def load_general_photo(self):
        """Загрузка общего фото с обработкой уже существующих тайлов"""
        file = filedialog.askopenfilename(
            title="Выберите общее фото",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.tiff *.tif"), ("All files", "*.*"))
        )
        if not file:
            return

        # Копируем общий файл во временную папку
        try:
            filename = os.path.basename(file)
            dest_path = os.path.join(self.temp_folder, filename)

            # Если файл уже существует, добавляем суффикс
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(original_dest_path)
                dest_path = f"{name}_{counter}{ext}"
                counter += 1

            shutil.copy2(file, dest_path)
            self.general_photo = dest_path
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать файл: {str(e)}")
            return

        # Проверяем, существует ли папка base и есть ли тайлы этого изображения
        if self.is_image_already_processed(self.general_photo):
            response = messagebox.askyesnocancel(
                "Изображение уже обработано",
                "Это изображение уже было нарезано.\n\n"
                "Вы хотите удалить старые изображения и заменить на новые?",
                icon=messagebox.QUESTION
            )

            if response is None:  # Отмена
                return
            elif response:  # Да - удалить и пересоздать
                self.delete_existing_tiles(self.general_photo)
            else:  # Нет - использовать существующие
                self.check_next_button()
                return

        # Нарезаем изображение (если не было выбора "использовать существующие")
        self.process_image(self.general_photo)
        self.check_next_button()

    def delete_existing_tiles(self, image_path):
        """Удаляет все тайлы для указанного изображения"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if not os.path.exists(self.tiles_folder):
            return

        deleted_count = 0
        for filename in os.listdir(self.tiles_folder):
            if filename.startswith(f"{image_name}_"):
                try:
                    os.remove(os.path.join(self.tiles_folder, filename))
                    deleted_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении файла {filename}: {e}")

        if deleted_count > 0:
            messagebox.showinfo("Успех", f"Удалено {deleted_count} старых тайлов изображения")

    def is_image_already_processed(self, image_path):
        """Проверяет, было ли изображение уже обработано"""
        if not os.path.exists(self.tiles_folder):
            return False

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # Проверяем, есть ли хотя бы один тайл этого изображения
        for file in os.listdir(self.tiles_folder):
            if file.startswith(f"{image_name}_"):
                return True
        return False

    def process_image(self, image_path):
        """Нарезка изображения на тайлы"""
        try:
            img = Image.open(image_path)
            width, height = img.size

            if not os.path.exists(self.tiles_folder):
                os.makedirs(self.tiles_folder)

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            tile_size = 256

            # Создаем окно прогресса
            progress_window = ProgressWindow(self.root, title="Нарезка изображения")
            self.root.update()  # Обновляем интерфейс

            # Вычисляем количество тайлов
            cols = (width + tile_size - 1) // tile_size  # Округление вверх
            rows = (height + tile_size - 1) // tile_size
            total_tiles = rows * cols
            processed = 0

            # Нарезаем изображение последовательно от левого верхнего угла
            for row in range(rows):  # row - номер строки (первая цифра)
                if hasattr(progress_window, 'cancelled') and progress_window.cancelled:
                    break
                for col in range(cols):  # col - номер столбца (вторая цифра)
                    x = col * tile_size
                    y = row * tile_size
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    tile = img.crop(box)

                    tile_path = os.path.join(
                        self.tiles_folder,
                        f"{image_name}_{row}_{col}.png"  # row_col - строка_столбец
                    )
                    tile.save(tile_path)

                    processed += 1
                    progress = int(processed / total_tiles * 100)
                    progress_window.progress["value"] = progress
                    progress_window.label.config(text=f"Обработано: {processed}/{total_tiles} тайлов")
                    progress_window.update()  # Обновляем окно прогресса

            progress_window.destroy()
            messagebox.showinfo("Успех", f"Изображение нарезано на {total_tiles} тайлов")

        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {str(e)}")

    def check_next_button(self):
        """Активируем кнопку Далее, если загружены оба типа фото"""
        if self.drone_photos and self.general_photo:
            self.next_btn.config(state=tk.NORMAL)

    def start_processing(self):
        """Начинаем обработку изображений в том же окне"""
        # Скрываем текущие элементы
        for widget in self.root.winfo_children():
            widget.pack_forget()

        # Создаем фрейм обработки в том же окне
        self.processing_frame = ProcessingFrame(self.root)
        self.processing_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)

        # Запускаем обработку в отдельном потоке
        self.processing_thread = threading.Thread(target=self.process_drone_photos)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_drone_photos(self):
        """Обработка фото дрона с использованием бэкенда"""
        try:
            # Импортируем функцию из бэкенда
            from backend import find_drone_photo_locations_sync

            # Функция обновления прогресса
            def progress_callback(message, percent):
                if hasattr(self, 'processing_frame') and self.processing_frame.winfo_exists():
                    self.root.after(0, lambda msg=message, pct=percent: self.processing_frame.update_progress(msg, pct))

            # Запускаем поиск с использованием бэкенда
            results = find_drone_photo_locations_sync(
                self.drone_photos,
                self.general_photo,
                self.tiles_folder,
                progress_callback
            )

            # Сохраняем результаты и создаем словарь номеров фото
            self.found_tiles = []
            self.photo_numbers = {}
            for i, (photo_path, row, col) in enumerate(results):
                self.found_tiles.append((row, col))
                self.photo_numbers[(row, col)] = i + 1  # Нумерация начинается с 1

            # После завершения обработки обновляем UI
            self.root.after(0, self.processing_complete)

        except ImportError as e:
            error_msg = f"Ошибка импорта бэкенда: {str(e)}"
            self.root.after(0, lambda: self.processing_error(error_msg))
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.processing_error(error_msg))

    def processing_complete(self):
        """Обработка завершена успешно"""
        # Удаляем фрейм обработки
        if hasattr(self, 'processing_frame'):
            self.processing_frame.destroy()

        # Показываем результаты
        self.create_results_window()

    def processing_error(self, error_message):
        """Обработка завершена с ошибкой"""
        # Удаляем фрейм обработки
        if hasattr(self, 'processing_frame'):
            self.processing_frame.destroy()

        # Показываем сообщение об ошибке и возвращаемся к первому окну
        messagebox.showerror("Ошибка", f"Ошибка при обработке: {error_message}")
        self.create_first_window()

    def create_results_window(self):
        """Создаем окно с результатами в том же окне"""
        self.clear_window()
        self.root.geometry("1300x800")

        # Основной контейнер
        main_container = ttk.Frame(self.root, style='Card.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Заголовок
        header = ttk.Label(main_container, text="Результаты анализа", style='Header.TLabel')
        header.pack(pady=(10, 20))

        # Основная область с результатами
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - фото с дрона с координатами
        left_panel = ttk.Frame(content_frame, style='Card.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        left_header = ttk.Label(left_panel, text="Фото с дрона", style='Title.TLabel')
        left_header.pack(pady=10)

        # Создаем прокручиваемую область для фото с дрона
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(canvas_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Отображаем фото с дрона и координаты
        for i, photo_path in enumerate(self.drone_photos):
            # Фрейм для каждого фото
            photo_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
            photo_frame.pack(fill=tk.X, pady=5, padx=5)

            # Отображаем фото
            try:
                img = Image.open(photo_path)
                img = img.resize((200, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                img_label = ttk.Label(photo_frame, image=photo)
                img_label.image = photo  # сохраняем ссылку
                img_label.pack()
            except Exception as e:
                ttk.Label(photo_frame, text="Ошибка загрузки").pack()

            # Отображаем координаты (пока троеточие, позже можно добавить реальные координаты)
            coord_frame = ttk.Frame(photo_frame)
            coord_frame.pack(fill=tk.X, pady=5)

            ttk.Label(coord_frame, text="Широта: ...").pack(anchor="w")
            ttk.Label(coord_frame, text="Долгота: ...").pack(anchor="w")

        # Правая панель - общее фото с возможностью зума
        right_panel = ttk.Frame(content_frame, style='Card.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        right_header = ttk.Label(right_panel, text="Общее фото", style='Title.TLabel')
        right_header.pack(pady=10)

        # Холст с возможностью зума и перемещения
        if self.general_photo:
            image_canvas = ZoomableMarkedImageCanvas(
                right_panel,
                self.general_photo,
                self.found_tiles,
                self.photo_numbers
            )
            image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Кнопка назад
        back_btn = ttk.Button(right_panel, text="Назад",
                              style='Primary.TButton',
                              command=self.create_first_window)
        back_btn.pack(pady=10)

    def clear_window(self):
        """Очищает окно от всех виджетов"""
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DronePhotoApp(root)
    root.mainloop()