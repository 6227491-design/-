
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Налаштування українських підписів
plt.rcParams["font.family"] = "DejaVu Sans"

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
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
except Exception:
    df = pd.read_csv(
        "imports-85.data",
        header=None,
        names=columns,
        na_values="?"
    )

features = [
    "horsepower", "curb-weight", "engine-size",
    "highway-mpg", "normalized-losses", "symboling"
]

for column in features + ["price"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

data = df[features + ["price"]].dropna()

print("Кількість автомобілів:", len(data))

# Завдання 1. Демонстрація перенавчання
print("\n=== ЗАВДАННЯ 1. ПЕРЕНАВЧАННЯ ===")

x = np.linspace(-3, 3, 35)
rng = np.random.default_rng(42)
y = np.sin(x) + rng.normal(0, 0.15, len(x))

x_train, x_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1), y,
    test_size=0.3,
    random_state=42
)

# Модель високого степеня для демонстрації перенавчання
model_over = make_pipeline(
    PolynomialFeatures(degree=20),
    LinearRegression()
)

model_over.fit(x_train, y_train)

train_pred = model_over.predict(x_train)
test_pred = model_over.predict(x_test)

print("Похибка навчання:",
      mean_squared_error(y_train, train_pred))
print("Похибка тестування:",
      mean_squared_error(y_test, test_pred))

x_plot = np.linspace(-3, 3, 500).reshape(-1, 1)

plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, label="Навчальні дані")
plt.scatter(x_test, y_test, label="Тестові дані")
plt.plot(
    x_plot,
    model_over.predict(x_plot),
    color="red",
    label="Модель високого степеня"
)
plt.ylim(-5, 5)
plt.xlabel("Незалежна змінна")
plt.ylabel("Залежна змінна")
plt.title("Завдання 1. Перенавчання моделі")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Перенавчання.png")
plt.show()


# Завдання 2. Демонстрація недонавчання
print("\n=== ЗАВДАННЯ 2. НЕДОНАВЧАННЯ ===")

# Використовуємо ті самі дані,
# але значно простішу лінійну модель
model_under = LinearRegression()
model_under.fit(x_train, y_train)

train_pred = model_under.predict(x_train)
test_pred = model_under.predict(x_test)

print("Похибка навчання:",
      mean_squared_error(y_train, train_pred))
print("Похибка тестування:",
      mean_squared_error(y_test, test_pred))

plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, label="Навчальні дані")
plt.scatter(x_test, y_test, label="Тестові дані")
plt.plot(
    x_plot,
    model_under.predict(x_plot),
    color="red",
    label="Лінійна модель"
)
plt.xlabel("Незалежна змінна")
plt.ylabel("Залежна змінна")
plt.title("Завдання 2. Недонавчання моделі")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Недонавчання.png")
plt.show()


# Завдання 3. Пошук гіперпараметрів Ridge Regression
print("\n=== ЗАВДАННЯ 3. RIDGE REGRESSION ===")

X = data[features]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Поліноміальні ознаки другого степеня,
# масштабування та Ridge Regression
pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(max_iter=100000)
)

# Значення параметра регуляризації
parameters = {
    "ridge__alpha": [
        0.001, 0.01, 0.1, 1, 10, 100, 1000
    ]
}

# Пошук оптимального параметра
grid = GridSearchCV(
    pipeline,
    parameters,
    cv=4,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

print("Найкращий параметр alpha:")
print(grid.best_params_)

print("Найкраща оцінка перехресної перевірки:")
print(-grid.best_score_)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print("MSE на тестових даних:",
      mean_squared_error(y_test, predictions))
print("R² на тестових даних:",
      r2_score(y_test, predictions))

# Графік порівняння фактичних і прогнозованих цін
plt.figure(figsize=(9, 6))
plt.scatter(y_test, predictions, label="Прогноз моделі")

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Ідеальний прогноз"
)

plt.xlabel("Фактична ціна автомобіля")
plt.ylabel("Прогнозована ціна автомобіля")
plt.title("Завдання 3. Ridge Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Ridge_Regression.png")
plt.show()

print("\nУсі три завдання виконано.")
