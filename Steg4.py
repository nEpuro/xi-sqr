from PIL import Image
import xlsxwriter
from scipy.stats import chi2
import os
import math
import matplotlib.pyplot as plt

def pixel_convert(x: int, y: int, img: Image.Image, color: str, bits: str):
    """
    Встраивает заданные биты в указанный цветовой канал пикселя.
    """
    try:
        r, g, b, a = img.getpixel((x, y))
    except ValueError:
        r, g, b = img.getpixel((x, y))
        a = None

    if color == 'b':
        modified = (b & (~((1 << len(bits)) - 1))) | int(bits, 2)
        b = modified
    elif color == 'g':
        modified = (g & (~((1 << len(bits)) - 1))) | int(bits, 2)
        g = modified
    elif color == 'r':
        modified = (r & (~((1 << len(bits)) - 1))) | int(bits, 2)
        r = modified
    else:
        return  # Неизвестный цветовой канал

    if a is not None:
        img.putpixel((x, y), (r, g, b, a))
    else:
        img.putpixel((x, y), (r, g, b))

def embed_text_in_image(blue_bit: int, green_bit: int, red_bit: int, original_path: str, text_path: str, output_path: str, percent: int):
    """
    Встраивает текст из файла text_path в изображение original_path и сохраняет результат в output_path.
    Процент встраивания указывает, какую долю доступных битов использовать для встраивания.
    """
    counter = 0
    img = Image.open(original_path)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    img = img.convert('RGB')
    img.save(output_path)
    img = Image.open(output_path)

    # Чтение текста из файла
    with open(text_path, "r", encoding='utf-8') as txt_file:
        text = txt_file.read()

    # Определение размера текста для встраивания
    total_available_bits = img.width * img.height * (blue_bit + green_bit + red_bit)
    size_text = int((total_available_bits / 8) * (percent / 100))
    if len(text) == 0:
        raise ValueError("Файл с текстом пуст.")
    text = (text * ((size_text // len(text)) + 1))[:size_text]

    # Преобразование текста в битовую строку
    bits = ''.join(format(x, '08b') for x in bytearray(text, 'utf-8'))

    # Встраивание битов в изображение
    for x in range(img.width):
        for y in range(img.height):
            if counter >= len(bits):
                break
            # Встраивание в синий канал
            bits_to_embed = bits[counter:counter + blue_bit].ljust(blue_bit, '0')
            pixel_convert(x, y, img, 'b', bits_to_embed)
            counter += blue_bit

            if counter >= len(bits):
                break
            # Встраивание в зеленый канал
            bits_to_embed = bits[counter:counter + green_bit].ljust(green_bit, '0')
            pixel_convert(x, y, img, 'g', bits_to_embed)
            counter += green_bit

            if counter >= len(bits):
                break
            # Встраивание в красный канал
            bits_to_embed = bits[counter:counter + red_bit].ljust(red_bit, '0')
            pixel_convert(x, y, img, 'r', bits_to_embed)
            counter += red_bit

    img.save(output_path)
    print(f"В изображение {output_path} успешно вставлен текст.")

def calculate_pixel_difference(original_path: str, modified_path: str):
    """
    Создаёт массив из значений пикселей до и после модификации.
    """
    img_original = Image.open(original_path)
    img_modified = Image.open(modified_path)
    result = []
    for x in range(img_original.width):
        for y in range(img_original.height):
            try:
                r_o, g_o, b_o, a_o = img_original.getpixel((x, y))
                r_m, g_m, b_m, a_m = img_modified.getpixel((x, y))
            except ValueError:
                r_o, g_o, b_o = img_original.getpixel((x, y))
                r_m, g_m, b_m = img_modified.getpixel((x, y))
            result.append([pixel_value(r_o, g_o, b_o), pixel_value(r_m, g_m, b_m)])
    return result

def calculate_pixel_difference_area(original_path: str, modified_path: str, x_block: int, y_block: int, size_area: int):
    """
    Создаёт массив из значений пикселей до и после модификации для заданной области.
    """
    img_original = Image.open(original_path)
    img_modified = Image.open(modified_path)
    result = []
    for x in range(x_block * size_area, min((x_block + 1) * size_area, img_original.width)):
        for y in range(y_block * size_area, min((y_block + 1) * size_area, img_original.height)):
            try:
                r_o, g_o, b_o, a_o = img_original.getpixel((x, y))
                r_m, g_m, b_m, a_m = img_modified.getpixel((x, y))
            except ValueError:
                r_o, g_o, b_o = img_original.getpixel((x, y))
                r_m, g_m, b_m = img_modified.getpixel((x, y))
            result.append([pixel_value(r_o, g_o, b_o), pixel_value(r_m, g_m, b_m)])
    return result

def pixel_value(r: int, g: int, b: int) -> float:
    """
    Вычисляет оттенок серого для пикселя.
    """
    return r * 0.299 + g * 0.587 + b * 0.114

def compute_max_difference(arr):
    """
    Находит максимальное абсолютное отклонение между старыми и новыми значениями пикселей.
    """
    return max(abs(old - new) for old, new in arr) if arr else 0

def compute_correlation(arr):
    """
    Вычисляет отношение (коэффициент корреляции).
    """
    numerator = sum(old * new for old, new in arr)
    denominator = sum(old**2 for old, new in arr)
    return (numerator / denominator) if denominator != 0 else 0

def compute_gssnr(arr):
    """
    Вычисляет отношение сигнал/шум (SNR) для массива пикселей.
    """
    if not arr:
        return 0, 0
    sum_sq_old = sum(old**2 for old, _ in arr)
    sum_old = sum(old for old, _ in arr)
    sum_sq_new = sum(new**2 for _, new in arr)
    sum_new = sum(new for _, new in arr)
    n = len(arr)
    mean_sq_old = sum_sq_old / n
    mean_old = (sum_old / n) ** 2
    mean_sq_new = sum_sq_new / n
    mean_new = (sum_new / n) ** 2
    sigma_old = math.sqrt(mean_sq_old - mean_old) if mean_sq_old > mean_old else 0
    sigma_new = math.sqrt(mean_sq_new - mean_new) if mean_sq_new > mean_new else 0
    return sigma_old, sigma_new

def gcd(a: int, b: int) -> int:
    """
    Вычисляет наибольший общий делитель (GCD) двух чисел.
    """
    while b:
        a, b = b, a % b
    return a

def calculate_gssnr(original_path: str, modified_path: str) -> float:
    """
    Вычисляет глобальное отношение сигнал/шум (SNR) для изображения.
    """
    segment_size = gcd_image_dimensions(original_path)
    img = Image.open(original_path)
    total_old_sq = 0
    total_diff_sq = 0
    num_segments_x = img.width // segment_size
    num_segments_y = img.height // segment_size

    for x in range(num_segments_x):
        for y in range(num_segments_y):
            area_pixels = calculate_pixel_difference_area(original_path, modified_path, x, y, segment_size)
            sigma_old, sigma_new = compute_gssnr(area_pixels)
            total_old_sq += sigma_old**2
            total_diff_sq += (sigma_old - sigma_new)**2

    return (total_old_sq / total_diff_sq) if total_diff_sq != 0 else 0

def gcd_image_dimensions(image_path: str) -> int:
    """
    Вычисляет наибольший общий делитель (GCD) ширины и высоты изображения.
    """
    img = Image.open(image_path)
    return gcd(img.width, img.height)

def chi_square_test(modified_path: str, color: str, size_block: int, num_block: int) -> float:
    """
    Вычисляет p-значение χ²-теста для заданного блока изображения.
    """
    img = Image.open(modified_path)
    color_pixel_count = [0] * 256
    counter = 0
    start = 1

    # Определение диапазона блоков для анализа
    x_start = (size_block * num_block) // img.height
    x_end = (size_block * (num_block + 1) + img.height - 1) // img.height

    for x in range(x_start, x_end):
        if counter > size_block * (num_block + 1):
            break
        y_start = (size_block * num_block * start) % img.height
        for y in range(y_start, img.height):
            start = 0
            if counter > size_block:
                break
            try:
                r, g, b, a = img.getpixel((x, y))
            except ValueError:
                r, g, b = img.getpixel((x, y))
            if color == 'r':
                color_pixel_count[r] += 1
            elif color == 'g':
                color_pixel_count[g] += 1
            elif color == 'b':
                color_pixel_count[b] += 1
            counter += 1

    # Подготовка частотных данных для χ²-теста
    frequency_color = []
    for k in range(0, len(color_pixel_count) // 2):
        observed = color_pixel_count[2 * k]
        expected = (color_pixel_count[2 * k] + color_pixel_count[2 * k + 1]) / 2
        if expected != 0:
            frequency_color.append([observed, expected])

    if len(frequency_color) <= 1:
        # Недостаточно данных для анализа
        return 0

    # Вычисление статистики χ²
    chi_sq_stat = 0
    for observed, expected in frequency_color:
        chi_sq_stat += ((observed - expected) ** 2) / expected

    degrees_of_freedom = len(frequency_color) - 1
    if degrees_of_freedom <= 0:
        return 0

    # Вычисление p-значения с использованием scipy
    p_value = 1 - chi2.cdf(chi_sq_stat, degrees_of_freedom)
    p_value = max(p_value, 0)  # Обеспечение неотрицательности
    p_value = 0 if p_value < 0.0001 else p_value

    return p_value

def create_analysis_tables(images_dir: str, text_file: str, blue_bit: int, green_bit: int, red_bit: int, analyze_color: str, percents: list, num_blocks: int = 128):
    """
    Создаёт модифицированные изображения, встраивает в них текст и проводит анализ.
    Результаты сохраняются в Excel-файлы.
    Также собирает данные для построения графиков.
    """
    # Генерация путей к оригинальным изображениям
    dirs_path = [os.path.join(images_dir, f"original_640x480_{i + 1}.png") for i in range(10)]

    img_analysis_results = {percent: [] for percent in percents}  # Словарь для хранения средних P-значений

    for i, original_path in enumerate(dirs_path):
        table_path = os.path.join(images_dir, f"table_{i + 1}.xlsx")
        workbook = xlsxwriter.Workbook(table_path)
        worksheet = workbook.add_worksheet()

        for p_idx, percent in enumerate(percents):
            modified_image_name = os.path.splitext(os.path.basename(original_path))[0] + f"_{percent}%.png"
            modified_path = os.path.join(images_dir, modified_image_name)

            # Встраивание текста в изображение
            embed_text_in_image(blue_bit, green_bit, red_bit, original_path, text_file, modified_path, percent)

            # Проведение анализа χ²-теста для 128 блоков
            p_values = []
            for j in range(num_blocks):
                p_val = chi_square_test(modified_path, analyze_color, (Image.open(modified_path).width * Image.open(modified_path).height) // num_blocks, j)
                if p_val > 0:
                    p_values.append(p_val)
                print(f"{modified_image_name} - Фрагмент {j + 1}: P = {p_val}")

            # Вычисление среднего арифметического P, учитывая только P > 0
            if p_values:
                average_p = sum(p_values) / len(p_values)
            else:
                average_p = 0
            img_analysis_results[percent].append(average_p)

            # Запись результатов в Excel
            header = f"{i + 1}_{percent}%"
            worksheet.write(0, p_idx, header)
            for row_idx, p_val in enumerate(p_values, start=1):
                worksheet.write(row_idx, p_idx, p_val)

        workbook.close()
        print(f"Таблица {table_path} готова.")

    # Создание финальной таблицы со средними значениями p-значений
    final_table_path = os.path.join(images_dir, "table_end.xlsx")
    workbook_final = xlsxwriter.Workbook(final_table_path)
    worksheet_final = workbook_final.add_worksheet()

    for p_idx, percent in enumerate(percents):
        for img_idx in range(len(dirs_path)):
            average_p = img_analysis_results[percent][img_idx]
            header = f"img{img_idx + 1}_{percent}%"
            worksheet_final.write(img_idx + 1, p_idx, average_p)  # Строка img_idx +1, столбец p_idx

    workbook_final.close()
    print(f"Финальная таблица {final_table_path} готова.")

    # Построение графиков
    plot_average_p_values(images_dir, img_analysis_results, dirs_path, percents)

def plot_average_p_values(images_dir: str, img_analysis_results: dict, dirs_path: list, percents: list):
    """
    Рисует три графика (для 25%, 50% и 65%) с названиями картинок по оси X и средними P по оси Y.
    """
    # Подготовка данных для графиков
    image_names = []
    for path in dirs_path:
        basename = os.path.basename(path)
        # Извлекаем номер картинки из имени файла
        # Предполагается формат: original_640x480_1.png
        try:
            image_number = os.path.splitext(basename)[0].split('_')[-1]
            label = f"Картинка {image_number} 640x480"
        except IndexError:
            label = basename  # В случае неожиданного формата имени
        image_names.append(label)

    for percent in percents:
        average_p = img_analysis_results[percent]
        plt.figure(figsize=(14, 7))
        bars = plt.bar(image_names, average_p, color='skyblue')
        plt.xlabel('Название картинки')
        plt.ylabel('Среднее арифметическое стегоанализа для каждой картинки')
        plt.title(f'Среднее арифметическое для стегоанализа Хи-квадрат при использовании {percent}% допустимого объёма встраивания текста')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(average_p) * 1.1 if average_p else 1)

        # Добавление значений над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        graph_path = os.path.join(images_dir, f"average_p_{percent}%.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"График для {percent}% сохранён как {graph_path}.")

if __name__ == '__main__':
    # Параметры
    DIR = "images"  # Директория с изображениями (без завершающего слэша)
    DIR_TXT = "input.txt"  # Путь к файлу с текстом для встраивания
    blue_bit = 1  # Количество используемых младших бит для синих каналов
    green_bit = 3  # Количество используемых младших бит для зеленых каналов
    red_bit = 3  # Количество используемых младших бит для красных каналов
    analyze_color = 'b'  # Цветовой канал для анализа (рекомендуется тот, где используется наименьшее количество бит)
    percents = [25, 50, 65]  # Проценты встраивания текста

    # Убедитесь, что директория существует
    if not os.path.isdir(DIR):
        print(f"Директория {DIR} не существует. Пожалуйста, создайте её и добавьте оригинальные изображения.")
    else:
        # Проверьте наличие оригинальных изображений
        missing_images = []
        for i in range(1, 11):
            image_path = os.path.join(DIR, f"original_640x480_{i}.png")
            if not os.path.isfile(image_path):
                missing_images.append(image_path)
        if missing_images:
            print("Отсутствуют следующие оригинальные изображения:")
            for img in missing_images:
                print(f" - {img}")
            print("Пожалуйста, добавьте все необходимые изображения перед запуском скрипта.")
        else:
            create_analysis_tables(DIR, DIR_TXT, blue_bit, green_bit, red_bit, analyze_color, percents)
