
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, f_oneway
from pathlib import Path

# Налаштування українського шрифту
plt.rcParams["font.family"] = "DejaVu Sans"

# Папка для збереження результатів
folder = Path("Результати_ЛР2")
folder.mkdir(exist_ok=True)

# Завантаження автомобільного набору даних
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/autos/imports-85.data"
)

columns = [
    "symboling", "normalized-losses", "make", "fuel-type",
    "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-type", "num-of-cylinders",
    "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]

try:
    df = pd.read_csv(
        url, header=None, names=columns, na_values="?"
    )
    print("Дані успішно завантажено.")
except Exception:
    df = pd.read_csv(
        "imports-85.data",
        header=None,
        names=columns,
        na_values="?"
    )
    print("Дані завантажено з локального файлу.")

numeric_columns = [
    "bore", "stroke", "compression-ratio",
    "horsepower", "wheel-base", "price"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Кількість автомобілів:", len(df))


# ЗАВДАННЯ 1
# Кореляція між чотирма характеристиками

print("\n=== ЗАВДАННЯ 1 ===")

features = [
    "bore", "stroke",
    "compression-ratio", "horsepower"
]

correlation = df[features].corr()

print("Матриця кореляції:")
print(correlation.round(3))

correlation.to_csv(
    folder / "Кореляція_завдання_1.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 6))
plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Коефіцієнт кореляції")
plt.xticks(range(4), features, rotation=30)
plt.yticks(range(4), features)

for i in range(4):
    for j in range(4):
        plt.text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Кореляція характеристик автомобілів")
plt.tight_layout()
plt.savefig(folder / "Завдання_1.png")
plt.show()


# ЗАВДАННЯ 2
# Кореляція між stroke та price

print("\n=== ЗАВДАННЯ 2 ===")

data2 = df[["stroke", "price"]].dropna()

r_stroke, p_stroke = pearsonr(
    data2["stroke"], data2["price"]
)

print("Коефіцієнт кореляції:", round(r_stroke, 4))
print("P-value:", p_stroke)

plt.figure(figsize=(9, 6))
plt.scatter(
    data2["stroke"],
    data2["price"],
    alpha=0.7
)

plt.xlabel("Хід поршня")
plt.ylabel("Ціна автомобіля")
plt.title("Залежність ціни від ходу поршня")
plt.grid(True)
plt.tight_layout()
plt.savefig(folder / "Завдання_2.png")
plt.show()


# ЗАВДАННЯ 3
# Перевірка наявності лінійної залежності

print("\n=== ЗАВДАННЯ 3 ===")

if abs(r_stroke) >= 0.7:
    print("Спостерігається сильна лінійна залежність.")
elif abs(r_stroke) >= 0.3:
    print("Спостерігається помірна лінійна залежність.")
else:
    print("Лінійна залежність слабка або майже відсутня.")

if p_stroke < 0.05:
    print("Кореляція статистично значуща.")
else:
    print("Кореляція статистично незначуща.")

print("Висновок зроблено на основі коефіцієнта Пірсона.")


# ЗАВДАННЯ 4
# Середня ціна автомобілів за типом кузова

print("\n=== ЗАВДАННЯ 4 ===")

grouped = (
    df.groupby("body-style")["price"]
    .agg(["mean", "count"])
    .sort_values("mean")
)

print("Середня ціна за типом кузова:")
print(grouped.round(2))

grouped.to_csv(
    folder / "Середня_ціна_за_кузовом.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 6))
plt.bar(
    grouped.index,
    grouped["mean"]
)

plt.xlabel("Тип кузова")
plt.ylabel("Середня ціна автомобіля")
plt.title("Середня ціна автомобілів за типом кузова")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(folder / "Завдання_4.png")
plt.show()


# ЗАВДАННЯ 5
# Кореляція Пірсона: wheel-base та price

print("\n=== ЗАВДАННЯ 5 ===")

data5 = df[["wheel-base", "price"]].dropna()

r5, p5 = pearsonr(
    data5["wheel-base"],
    data5["price"]
)

print("Коефіцієнт Пірсона:", round(r5, 4))
print("P-value:", p5)

if p5 < 0.05:
    print("Кореляція статистично значуща.")
else:
    print("Кореляція статистично незначуща.")


# ЗАВДАННЯ 6
# Кореляція Пірсона: horsepower та price

print("\n=== ЗАВДАННЯ 6 ===")

data6 = df[["horsepower", "price"]].dropna()

r6, p6 = pearsonr(
    data6["horsepower"],
    data6["price"]
)

print("Коефіцієнт Пірсона:", round(r6, 4))
print("P-value:", p6)

if p6 < 0.05:
    print("Кореляція статистично значуща.")
else:
    print("Кореляція статистично незначуща.")


# ЗАВДАННЯ 7
# Дисперсійний аналіз ANOVA

print("\n=== ЗАВДАННЯ 7 ===")

# Порівняння цін за типами кузова
groups_body = [
    group["price"].dropna().values
    for _, group in df.groupby("body-style")
    if group["price"].notna().sum() >= 2
]

f_body, p_body = f_oneway(*groups_body)

print("\nДисперсійний аналіз за типом кузова:")
print("F-критерій:", round(f_body, 4))
print("P-value:", p_body)

# Порівняння цін за типами приводу
groups_drive = [
    group["price"].dropna().values
    for _, group in df.groupby("drive-wheels")
    if group["price"].notna().sum() >= 2
]

f_drive, p_drive = f_oneway(*groups_drive)

print("\nДисперсійний аналіз за типом приводу:")
print("F-критерій:", round(f_drive, 4))
print("P-value:", p_drive)

# Порівняння цін за типами двигуна
groups_engine = [
    group["price"].dropna().values
    for _, group in df.groupby("engine-type")
    if group["price"].notna().sum() >= 2
]

f_engine, p_engine = f_oneway(*groups_engine)

print("\nДисперсійний аналіз за типом двигуна:")
print("F-критерій:", round(f_engine, 4))
print("P-value:", p_engine)

print("\n=== ВИСНОВКИ ДИСПЕРСІЙНОГО АНАЛІЗУ ===")

analyses = {
    "Тип кузова": p_body,
    "Тип приводу": p_drive,
    "Тип двигуна": p_engine
}

for name, p_value in analyses.items():
    if p_value < 0.05:
        print(
            f"{name}: виявлено статистично значущі "
            "відмінності між середніми цінами."
        )
    else:
        print(
            f"{name}: статистично значущих "
            "відмінностей не виявлено."
        )

# Збереження підсумкової таблиці
summary = pd.DataFrame({
    "Показник": [
        "Хід поршня та ціна",
        "Колісна база та ціна",
        "Потужність та ціна"
    ],
    "Коефіцієнт Пірсона": [
        r_stroke, r5, r6
    ],
    "P-value": [
        p_stroke, p5, p6
    ]
})

summary.to_csv(
    folder / "Підсумкові_результати.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nЛабораторну роботу №2 виконано.")
print("Результати збережено в папці:", folder.resolve())
