
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Налаштування українського шрифту для графіків
plt.rcParams["font.family"] = "DejaVu Sans"

# Папка для збереження результатів
OUTPUT = Path("Результати_лабораторної_роботи")
OUTPUT.mkdir(exist_ok=True)

# Назви стовпців автомобільного набору даних
columns = [
    "symboling", "normalized-losses", "make", "fuel-type",
    "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-type", "num-of-cylinders",
    "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]

url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/autos/imports-85.data"
)

# Завантаження даних
try:
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
    print("Дані успішно завантажено з інтернету.")
except Exception:
    df = pd.read_csv(
        "imports-85.data",
        header=None,
        names=columns,
        na_values="?"
    )
    print("Дані завантажено з локального файлу.")

# Підготовка даних
features = ["engine-size", "normalized-losses", "highway-mpg"]

for col in features + ["price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

data = df[features + ["price"]].dropna().copy()

print("\nКількість записів:", len(data))
print("\nСтатистичні характеристики:")
print(data.describe().round(2))

# Розподіл даних на навчальну та тестову вибірки
train, test = train_test_split(
    data, test_size=0.2, random_state=42
)

results = []

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nНазва моделі: {name}")
    print("Середньоквадратична похибка (MSE):", round(mse, 2))
    print("Коефіцієнт детермінації (R²):", round(r2, 4))

    results.append({
        "Модель": name,
        "MSE": mse,
        "R²": r2
    })

    return predictions


# ЗАВДАННЯ 1
# Побудова лінійної регресії
print("\n=== ЗАВДАННЯ 1. ЛІНІЙНА РЕГРЕСІЯ ===")

model1 = LinearRegression()

X1_train = train[["engine-size"]]
X1_test = test[["engine-size"]]

pred1 = evaluate(
    "Лінійна регресія",
    model1,
    X1_train, X1_test,
    train["price"], test["price"]
)

print("Коефіцієнт регресії:", model1.coef_[0])
print("Вільний член:", model1.intercept_)

x_line = np.linspace(
    data["engine-size"].min(),
    data["engine-size"].max(),
    200
)

plt.figure(figsize=(9, 6))
plt.scatter(
    data["engine-size"],
    data["price"],
    alpha=0.6,
    label="Фактичні дані"
)

plt.plot(
    x_line,
    model1.predict(
        pd.DataFrame({"engine-size": x_line})
    ),
    color="red",
    label="Лінійна модель"
)

plt.xlabel("Робочий об'єм двигуна")
plt.ylabel("Ціна автомобіля")
plt.title("Завдання 1. Лінійна регресія")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "Завдання_1.png")
plt.show()


# ЗАВДАННЯ 2
# Побудова множинної лінійної регресії
print("\n=== ЗАВДАННЯ 2. МНОЖИННА РЕГРЕСІЯ ===")

multiple_features = ["normalized-losses", "highway-mpg"]

model2 = LinearRegression()

pred2 = evaluate(
    "Множинна лінійна регресія",
    model2,
    train[multiple_features],
    test[multiple_features],
    train["price"],
    test["price"]
)

print("Коефіцієнти регресії:", model2.coef_)
print("Вільний член:", model2.intercept_)

plt.figure(figsize=(9, 6))

plt.scatter(
    test["price"],
    pred2,
    alpha=0.7,
    label="Прогнозовані значення"
)

min_price = min(test["price"].min(), pred2.min())
max_price = max(test["price"].max(), pred2.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    "r--",
    label="Ідеальний прогноз"
)

plt.xlabel("Фактична ціна автомобіля")
plt.ylabel("Прогнозована ціна автомобіля")
plt.title("Завдання 2. Множинна лінійна регресія")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "Завдання_2.png")
plt.show()


# ЗАВДАННЯ 3
# Поліноміальна регресія 11-го порядку
print("\n=== ЗАВДАННЯ 3. ПОЛІНОМІАЛЬНА РЕГРЕСІЯ ===")

# Створення демонстраційних даних
rng = np.random.default_rng(42)

x = np.linspace(-3, 3, 60)
y = 0.5 * x**3 - 2 * x**2 + 3 * x + 5
y = y + rng.normal(0, 3, size=len(x))

X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model3 = make_pipeline(
    PolynomialFeatures(degree=11),
    LinearRegression()
)

pred3 = evaluate(
    "Поліноміальна регресія 11-го порядку",
    model3,
    X_train, X_test,
    y_train, y_test
)

x_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
y_plot = model3.predict(x_plot)

plt.figure(figsize=(9, 6))
plt.scatter(
    x, y,
    label="Початкові дані"
)

plt.plot(
    x_plot, y_plot,
    color="red",
    label="Поліном 11-го порядку"
)

plt.xlabel("Незалежна змінна x")
plt.ylabel("Залежна змінна y")
plt.title("Завдання 3. Поліноміальна регресія")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "Завдання_3.png")
plt.show()


# ЗАВДАННЯ 4
# Порівняння побудованих моделей
print("\n=== ЗАВДАННЯ 4. ПОРІВНЯННЯ МОДЕЛЕЙ ===")

summary = pd.DataFrame(results)

print("\nРезультати оцінювання моделей:")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    OUTPUT / "Результати_моделей.csv",
    index=False,
    encoding="utf-8-sig"
)

# Порівнюємо перші дві моделі на однакових даних
best = min(results[:2], key=lambda item: item["MSE"])

print("\nНайкраща модель серед перших двох за MSE:")
print(best["Модель"])

print("\n=== ВИСНОВКИ ===")
print("1. Побудовано лінійну регресійну модель залежності ціни автомобіля від об'єму двигуна.")
print("2. Створено модель множинної лінійної регресії з двома незалежними змінними.")
print("3. Побудовано поліноміальну регресійну модель 11-го порядку.")
print("4. Якість моделей оцінено за показниками MSE та R².")
print("5. Менше значення MSE свідчить про точніший прогноз на однаковій тестовій вибірці.")

print("\nУсі результати збережено в папці:")
print(OUTPUT.resolve())

