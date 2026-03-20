#!/usr/bin/env python3
"""
demo.py — демонстрация всех компонентов Depersonalizer.
Запуск: python demo.py
"""

import warnings

warnings.filterwarnings("ignore")

from depersonalizer import (
    Depersonalizer,
    DifferentialPrivacyProcessor,
    KAnonymityProcessor,
    _DOCX_OK,
    _NATASHA_OK,
    _PANDAS_OK,
    _PRESIDIO_OK,
    _PYMUPDF_OK,
)

SEP = "─" * 60


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════
# 1. REGEX — только встроенные паттерны (без внешних NER-либ)
# ══════════════════════════════════════════════════════════════
section("1. REGEX-ДЕТЕКТОР  (без внешних зависимостей)")

samples = [
    "Менеджер Иванов Иван Петрович, ivan.ivanov@sber.ru, тел. +7-900-123-45-67",
    "Паспорт 4510 123456 выдан 15.06.2010. СНИЛС: 123-456-789 01.",
    "Карта клиента: 4276 1234 5678 9012. ИНН физ. лица: 770100123456.",
    "IP: 192.168.1.100 | сайт: https://lk.sberbank.ru/profile?id=42",
    "Дата рождения: 01.01.1990. IBAN: RU0204452560040702810412345678901.",
]

dp_regex = Depersonalizer(
    mode="placeholder",
    use_natasha=False,
    use_presidio=False,
)

for text in samples:
    result = dp_regex.anonymize_text(text)
    print(f"\nДО:    {text}")
    print(f"ПОСЛЕ: {result}")

report = dp_regex.get_report()
print(f"\n{SEP}")
print(f"Обнаружено уникальных сущностей: {report['total_entities']}")
print("Маппинг:")
for orig, ph in report["mapping"].items():
    print(f"  {ph:<22}  <-  {orig!r}")

# Режим HASH
section("1b. РЕЖИМ HASH")
dp_hash = Depersonalizer(mode="hash", use_natasha=False, use_presidio=False)
for text in samples[:2]:
    print(f"\nДО:    {text}")
    print(f"ПОСЛЕ: {dp_hash.anonymize_text(text)}")

# Режим MASK
section("1c. РЕЖИМ MASK (█)")
dp_mask = Depersonalizer(mode="mask", use_natasha=False, use_presidio=False)
for text in samples[:2]:
    print(f"\nДО:    {text}")
    print(f"ПОСЛЕ: {dp_mask.anonymize_text(text)}")


# ══════════════════════════════════════════════════════════════
# 2. NATASHA NER (если установлена)
# ══════════════════════════════════════════════════════════════
section("2. NATASHA NER  (RU: PER / LOC / ORG)")

if _NATASHA_OK:
    from depersonalizer import NatashaDetector

    nat = NatashaDetector()
    nat_samples = [
        "Директор Петров Сергей Александрович прибыл в Москву из Санкт-Петербурга.",
        "Сотрудник ОАО «Газпром» Иванова Мария обратилась в МФЦ.",
    ]
    for text in nat_samples:
        matches = nat.detect(text)
        print(f"\nТекст: {text}")
        for m in matches:
            print(f"  [{m.source}] {m.entity_type:<6} -> {m.original_text!r}")
else:
    print("  ⚠ natasha не установлена. Запустите: pip install natasha")


# ══════════════════════════════════════════════════════════════
# 3. PRESIDIO (если установлен)
# ══════════════════════════════════════════════════════════════
section("3. PRESIDIO  (мультиязычный NER)")

if _PRESIDIO_OK:
    try:
        from depersonalizer import PresidioDetector

        pres   = PresidioDetector(language="ru")
        p_text = "Иванов Иван, дата рождения 15.06.1990, тел. +79001234567."
        matches = pres.detect(p_text)
        print(f"\nТекст: {p_text}")
        for m in matches:
            print(f"  [{m.source}] {m.entity_type:<20} conf={m.confidence:.2f} -> {m.original_text!r}")
    except RuntimeError as e:
        print(f"  ⚠ {e}")
else:
    print("  ⚠ presidio не установлена.")
    print("    pip install presidio-analyzer presidio-anonymizer spacy")
    print("    python -m spacy download ru_core_news_sm")


# ══════════════════════════════════════════════════════════════
# 4. АНСАМБЛЬ (Regex + Natasha + Presidio)
# ══════════════════════════════════════════════════════════════
section("4. АНСАМБЛЬ-ДЕТЕКТОР  (Regex + NER)")

dp_full = Depersonalizer(
    mode="placeholder",
    use_natasha=_NATASHA_OK,
    use_presidio=_PRESIDIO_OK,
)
ensemble_text = (
    "Клиент Петрова Анна Сергеевна (anna@mail.ru) обратилась "
    "с паспортом 4510 654321 по тел. 8-800-555-35-35 из г. Казань."
)
result = dp_full.anonymize_text(ensemble_text)
print(f"\nДО:    {ensemble_text}")
print(f"ПОСЛЕ: {result}")


# ══════════════════════════════════════════════════════════════
# 5. K-АНОНИМНОСТЬ
# ══════════════════════════════════════════════════════════════
section("5. K-АНОНИМНОСТЬ  (k=2)")

if _PANDAS_OK:
    import pandas as pd

    # Данные с намеренными дублями в QI-диапазонах (возраст, зип, год)
    # после обобщения: (20-29, 101***, 199x) встречается >=2 раза
    df = pd.DataFrame({
        "name": [
            "Иванов И.", "Петров П.", "Сидоров С.",
            "Козлов К.", "Новиков Н.", "Смирнов С.", "Орлов О.",
        ],
        "age": [25, 27, 23, 55, 58, 31, 33],
        "zip_code": [
            "101000", "101001", "101002",
            "191000", "191001", "190001", "190002",
        ],
        "birth_date": [
            "01.01.1999", "05.03.1997", "10.07.1999",
            "20.07.1969", "15.09.1966", "12.11.1993", "22.04.1991",
        ],
        "salary": [80_000, 90_000, 75_000, 200_000, 210_000, 120_000, 130_000],
    })

    print("\nИсходный DataFrame:")
    print(df.to_string(index=False))
    print(
        "\nОжидание: после обобщения age->range, zip->3 chars, birth->year\n"
        "  (25,27,23) -> 20-29 | (101xxx) -> 101*** | (1999,1997,1999) -> 1999/1997\n"
        "  (55,58) -> 50-59 | (191xxx) -> 191*** | (1969,1966) -> 1969/1966\n"
        "  Группы с count>=2 будут сохранены."
    )

    kp      = KAnonymityProcessor(k=2)
    df_anon = kp.process(
        df,
        quasi_ids=["age", "zip_code", "birth_date"],
        strategies={
            "age":        "age_range",
            "zip_code":   "zip_prefix",
            "birth_date": "year_only",
        },
    )
    print("\nПосле K-анонимности (k=2):")
    print(df_anon.to_string(index=False))

    # ── 6. Дифференциальная приватность ──────────────────────
    section("6. ДИФФЕРЕНЦИАЛЬНАЯ ПРИВАТНОСТЬ  (e=1.0, Лаплас, sens=20000)")

    # sensitivity = типичный шаг изменения зарплаты в наборе (20 000 руб.)
    # e=1.0 → scale = 20000 / 1.0 = 20000 → шум ≈ ±20k к каждой зарплате
    dp_proc = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=20_000)

    if df_anon.empty:
        # Если k-анонимность всё подавила — показываем на исходном df
        df_for_dp = df.copy()
        print("  (используем исходный df, т.к. k-анонимность подавила все записи)")
    else:
        df_for_dp = df_anon.copy()

    df_dp = dp_proc.add_noise(df_for_dp, numeric_columns=["salary"])
    print(df_dp[["name", "salary"]].to_string(index=False))

    print(
        f"\nПриватизированный счётчик обращений (100): "
        f"{DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1).privatize_count(100)}"
    )

else:
    print("  ⚠ pandas не установлен: pip install pandas")


# ══════════════════════════════════════════════════════════════
# 7. ФАЙЛОВЫЙ ПРОЦЕССИНГ (TXT + опционально PDF/DOCX)
# ══════════════════════════════════════════════════════════════
section("7. ФАЙЛОВЫЙ ПРОЦЕССИНГ")

import tempfile
from pathlib import Path

txt_content = (
    "Заявление от Иванова Ивана Ивановича.\n"
    "Телефон: +7-916-123-45-67\n"
    "Email: ivan@example.com\n"
    "Паспорт: 4510 123456\n"
    "ИНН: 770100123456\n"
)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp     = Path(tmpdir)
    src_txt = tmp / "test_input.txt"
    src_txt.write_text(txt_content, encoding="utf-8")

    dp_file  = Depersonalizer(mode="placeholder", use_natasha=False, use_presidio=False)
    out_txt  = dp_file.anonymize_file(src_txt)
    print(f"\nTXT (ДО):\n{txt_content}")
    print(f"TXT (ПОСЛЕ):\n{out_txt.read_text(encoding='utf-8')}")

print(f"PDF-процессор:  {'✓ pymupdf установлен' if _PYMUPDF_OK else '✗ pip install pymupdf'}")
print(f"DOCX-процессор: {'✓ python-docx установлен' if _DOCX_OK else '✗ pip install python-docx'}")

print(f"\n{'═' * 60}")
print("  ✓ Demo завершён.")
print(f"{'═' * 60}")
