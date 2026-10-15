# Titanic Survival Classification

Финальный проект по бинарной классификации пассажиров Titanic. В проекте последовательно проверены классические модели, три реализации градиентного бустинга, полносвязная нейронная сеть, основные варианты feature engineering и ансамбли.

Финальная воспроизводимая модель — hard voting из Random Forest, CatBoost и XGBoost. Она получила `0.8440 ± 0.0110` на 5-fold Stratified CV и `0.77272` на Public Leaderboard Kaggle.

## Быстрый запуск

Из каталога `final_proj`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

На Windows активация окружения выполняется командой:

```powershell
.venv\Scripts\activate
```

Один запуск `main.py`:

1. загружает `titanic/train.csv` и `titanic/test.csv`;
2. выполняет 5-fold Stratified CV для компонентов и ансамблей;
3. сохраняет таблицу метрик в `results/final_cv_results.csv`;
4. обучает финальные модели на всей обучающей выборке;
5. создаёт `submissions/submission_hard_ensemble.csv`.

Поведение запуска и гиперпараметры меняются в `config.py`. Для быстрого инференса без повторной кросс-валидации установите `RUN_CROSS_VALIDATION = False`.

## Данные и признаки

Используются семь исходных признаков:

- числовые: `Age`, `Fare`, `SibSp`, `Parch`;
- категориальные: `Pclass`, `Sex`, `Embarked`.

Для Random Forest числовые пропуски заполняются медианой с дополнительным индикатором пропуска, категориальные — наиболее частым значением, затем применяется one-hot encoding. CatBoost и XGBoost получают категориальные признаки в нативном формате.

Проверенные `FamilySize`, `IsAlone`, `Title` и `CabinDeck` не улучшили средний CV Random Forest, поэтому в финальный набор не вошли.

## Валидация и результаты

Во всех итоговых сравнениях используется один и тот же `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` и метрика accuracy.

| Модель | Mean CV accuracy | CV std | Public LB |
|---|---:|---:|---:|
| Logistic Regression baseline | 0.7946 | 0.0182 | 0.77033 |
| KNN | 0.8159 | 0.0173 | — |
| Decision Tree | 0.8384 | 0.0131 | — |
| Random Forest | 0.8440 | 0.0235 | — |
| CatBoost | 0.8372 | 0.0082 | — |
| LightGBM | 0.8406 | 0.0185 | — |
| XGBoost | 0.8440 | 0.0207 | — |
| DNN, оценка по трём seed | 0.8268 | 0.0186 | — |
| Soft ensemble | 0.8440 | 0.0200 | — |
| **Hard ensemble** | **0.8440** | **0.0110** | **0.77272** |
| Stacking | 0.8417 | 0.0194 | — |

Hard voting не увеличил среднюю CV accuracy относительно Random Forest и XGBoost, но более чем вдвое уменьшил разброс Random Forest между фолдами. Поэтому он выбран как наиболее устойчивый воспроизводимый вариант.

В каталоге `submissions` также сохранён ранний файл `submission.csv` со score `0.77511`. Он не считается финальным артефактом, поскольку код его создания не сохранился. Официальный результат проекта — `submission_hard_ensemble.csv`.

## EDA

Ноутбук `EDA.ipynb` содержит:

- распределение таргета;
- статистики и пропуски train/test;
- анализ категориальных и числовых признаков;
- распределения и выбросы;
- корреляционную матрицу и визуализации.

Основные наблюдения: целевой класс несбалансирован умеренно, сильнее всего выживание связано с полом и классом билета, `Age` и особенно `Cabin` содержат пропуски, а `Fare` имеет выраженный правый хвост.

## Структура проекта

```text
final_proj/
├── main.py                 # воспроизводимый CV, обучение и inference
├── config.py               # пути, признаки и гиперпараметры
├── pipeline.py             # общая sklearn-предобработка
├── requirements.txt
├── EDA.ipynb
├── experiments/            # последовательность exp01–exp13
├── info/                   # журнал, алгоритм и чеклист
├── results/                # итоговая таблица CV
├── submissions/            # CSV для Kaggle
└── titanic/                # train.csv и test.csv
```

## Дополнительные эксперименты

- `experiments/exp08_dnn.py` — DNN на PyTorch;
- `experiments/exp09_feature_engineering.py` — основные engineered-признаки;
- `experiments/exp10_voting.py` — soft voting классических моделей;
- `experiments/exp11_strong_ensemble.py` — soft и hard voting сильных моделей;
- `experiments/exp12_stacking.py` — stacking с Logistic Regression;
- `experiments/exp13_final_ensemble.py` — только финальное обучение и submission.

