import pandas as pd
import numpy as np

# Лабораторна робота №1. Data Wrangling
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
headers = [
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
    'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
    'highway-mpg', 'price'
]

df = pd.read_csv(url, names=headers, na_values='?')
print('Початкові дані:')
print(df.head().to_string())
print('\nКількість пропущених значень до обробки:')
print(df.isna().sum().to_string())

# Завдання 1: заміна NaN у stroke середнім
# Перетворюємо числові стовпці, які могли бути прочитані як текст.
text_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
                'body-style', 'drive-wheels', 'engine-location',
                'engine-type', 'num-of-cylinders', 'fuel-system']
for col in df.columns:
    if col not in text_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

stroke_mean = df['stroke'].mean()
df['stroke'] = df['stroke'].fillna(stroke_mean)
print(f'\nЗавдання 1. Середнє stroke: {stroke_mean:.4f}')
print('Пропуски у stroke:', df['stroke'].isna().sum())

# Завдання 2: нормалізація height до діапазону [0, 1]
height_min, height_max = df['height'].min(), df['height'].max()
df['height'] = (df['height'] - height_min) / (height_max - height_min)
print('\nЗавдання 2. Нормалізований height:')
print(df['height'].head().to_string(index=False))

# Завдання 3.1 і 3.2: індикаторні змінні для aspiration
aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration', dtype=int)
df = pd.concat([df.drop(columns=['aspiration']), aspiration_dummies], axis=1)
print('\nЗавдання 3. Індикаторні змінні aspiration:')
print(aspiration_dummies.head().to_string(index=False))

# Завдання 4: заповнення решти пропусків
# Для числових даних — середнє, для категоріальних — мода.
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

print('\nЗавдання 4. Пропуски після обробки:')
print(df.isna().sum().to_string())
print('\nПідсумкова таблиця (перші 5 рядків):')
print(df.head().to_string())

df.to_csv('lab1_result.csv', index=False, encoding='utf-8-sig')
print('\nГотово! Результат збережено у lab1_result.csv')
