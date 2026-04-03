#!/usr/bin/env python3
"""
Пример использования деперсонализации изображений.

Требования:
1. Установить Tesseract OCR:
   - macOS: brew install tesseract tesseract-lang
   - Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-rus
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

2. Установить Python библиотеки:
   pip install Pillow pytesseract
"""

from depersonalizer import Depersonalizer, _PILLOW_OK, _TESSERACT_OK

# Проверка зависимостей
if not _PILLOW_OK or not _TESSERACT_OK:
    print("Обработка изображений недоступна.")
    print("Установите зависимости:")
    print("  pip install Pillow pytesseract")
    print("  + Tesseract OCR engine (смотрите TESSERACT_SETUP.md)")
else:
    print("Обработка изображений доступна")

    # Использование
    dp = Depersonalizer(mode="placeholder")

    # Обработка PNG файла
    print("\nОбработка screenshot.png...")
    try:
        dp.anonymize_file("screenshot.png", "screenshot_anon.png")
        print("Готово: screenshot_anon.png")
    except FileNotFoundError:
        print("Файл screenshot.png не найден")
    except Exception as e:
        print(f"Ошибка: {e}")

    # Обработка JPG файла
    print("\nОбработка photo.jpg...")
    try:
        dp.anonymize_file("photo.jpg", "photo_anon.jpg")
        print("Готово: photo_anon.jpg")
    except FileNotFoundError:
        print("Файл photo.jpg не найден")
    except Exception as e:
        print(f"Ошибка: {e}")

print("\n---")
print("Поддерживаемые форматы:")
print("  - PNG (.png)")
print("  - JPEG (.jpg, .jpeg)")
print("  - Другие форматы, поддерживаемые Pillow")
