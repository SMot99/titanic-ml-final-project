# Titanic Final Project — журнал экспериментов

Последнее обновление: 2026-10-15

## Статус

- [x] EDA
- [x] Общая предобработка
- [x] Stratified 5-fold CV
- [x] Линейные модели, KNN и деревья
- [x] CatBoost, LightGBM и XGBoost
- [x] DNN на PyTorch
- [x] Основной feature engineering
- [x] Averaging, voting и stacking
- [x] Финальная модель и Kaggle submission
- [x] `main.py`, `config.py`, зависимости и README

## Валидация

Основная метрика — accuracy. Для итоговых сравнений используется:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Предобработка обучается заново внутри каждого фолда. Числовые пропуски заполняются медианой с индикатором пропуска; категориальные — наиболее частым значением. Для sklearn-моделей категории кодируются one-hot. Масштабирование применяется только там, где оно влияет на алгоритм.

## Основные результаты

| Подход | Лучшие параметры | Mean CV | CV std | Public LB |
|---|---|---:|---:|---:|
| Logistic Regression baseline | median + indicator, mode, one-hot, scaling | 0.7946 | 0.0182 | 0.77033 |
| Logistic Regression, L2 | `C=0.1` | 0.8025 | 0.0184 | — |
| Logistic Regression, L1 | `C=1.0` | 0.7946 | 0.0182 | — |
| Logistic Regression, Elastic Net | `C=0.1`, `l1_ratio=0.25` | 0.7991 | 0.0197 | — |
| KNN | `n_neighbors=7`, uniform, Manhattan | 0.8159 | 0.0173 | — |
| Decision Tree | `max_depth=8`, `min_samples_leaf=5` | 0.8384 | 0.0131 | — |
| Random Forest | 100 trees, depth 6, `max_features=None` | 0.8440 | 0.0235 | — |
| CatBoost | 300 iterations, depth 4, lr 0.1, L2 5 | 0.8372 | 0.0082 | — |
| LightGBM | 300 trees, depth 9, 15 leaves, lr 0.05 | 0.8406 | 0.0185 | — |
| XGBoost | 300 trees, depth 3, lr 0.1, alpha/lambda 1 | 0.8440 | 0.0207 | — |
| DNN, три seed | 13→32→1, ReLU, dropout 0.1, AdamW | 0.8268 | 0.0186 | — |
| Soft ensemble | среднее вероятностей RF + CatBoost + XGBoost | 0.8440 | 0.0200 | — |
| **Hard ensemble** | majority vote RF + CatBoost + XGBoost | **0.8440** | **0.0110** | **0.77272** |
| Stacking | три сильные модели + Logistic Regression | 0.8417 | 0.0194 | — |

## DNN

Базовая архитектура: `input → 32 → 1`, ReLU, `BCEWithLogitsLoss`. Проверены 50/75/100 эпох, ширина 16/32/64, два скрытых слоя, dropout 0/0.1/0.2, BatchNorm, ReLU/LeakyReLU/Tanh, learning rate 0.0005/0.001/0.003, batch size 16/32/64, Adam/AdamW и StepLR.

| Вариант | Mean CV | CV std |
|---|---:|---:|
| ReLU, dropout 0.1, Adam | 0.8283 | 0.0194 |
| BatchNorm | 0.8226 | 0.0163 |
| LeakyReLU | 0.8249 | 0.0206 |
| Tanh | 0.8036 | 0.0182 |
| AdamW, `weight_decay=0.01`, лучший одиночный seed | 0.8305 | 0.0212 |
| AdamW, оценка по трём seed | 0.8268 | 0.0186 |
| StepLR | 0.8193 | 0.0213 |

Вывод: DNN работает устойчиво, но на маленьких табличных данных уступает деревьям и бустингам.

## Feature engineering

| Набор признаков для Random Forest | Mean CV | CV std |
|---|---:|---:|
| Исходные признаки | **0.8440** | 0.0235 |
| `FamilySize`, `IsAlone` | 0.8395 | 0.0191 |
| `Title` | 0.8428 | 0.0200 |
| `CabinDeck` | 0.8350 | 0.0296 |
| Все четыре engineered-признака | 0.8395 | 0.0254 |

Новые признаки не повысили среднюю accuracy. Финальная модель использует исходные `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`, `Sex`, `Embarked`.

## Ансамбли

| Ансамбль | Mean CV | CV std |
|---|---:|---:|
| Soft voting: Logistic Regression + KNN + Random Forest | 0.8227 | 0.0233 |
| Weighted soft voting, веса 1:1:3 | 0.8350 | 0.0210 |
| Soft: Random Forest + CatBoost + XGBoost | 0.8440 | 0.0200 |
| **Hard: Random Forest + CatBoost + XGBoost** | **0.8440** | **0.0110** |
| Stacking с Logistic Regression | 0.8417 | 0.0194 |

Hard voting выбран за минимальный разброс при сохранении лучшей средней accuracy.

## Финальная модель

| Модель | CV | Public LB | Private LB |
|---|---:|---:|---:|
| Hard ensemble | 0.8440 | 0.77272 | — |

Финальный воспроизводимый файл: `submissions/submission_hard_ensemble.csv`.

В каталоге также находится ранний `submission.csv` с Public LB `0.77511`, но код его создания не сохранился, поэтому он не считается итоговым артефактом проекта.

## Обработанные идеи

### Полезные

- единая предобработка без утечки внутри CV;
- L2-регуляризация Logistic Regression;
- неглубокие деревья и бустинги;
- hard voting сильных разнородных моделей;
- повторная проверка DNN с несколькими seed.

### Нейтральные

- `Title` немного снизил mean CV, но сделал Random Forest стабильнее;
- soft averaging сильных моделей сохранил mean CV, но не превзошёл компоненты;
- AdamW дал небольшое улучшение на одном seed, которое уменьшилось при повторной проверке.

### Неудачные

- равные веса слабых и сильных моделей в voting;
- BatchNorm и Tanh в DNN;
- StepLR с уменьшением шага каждые 25 эпох;
- `CabinDeck` из-за большого числа пропусков;
- полный набор engineered-признаков.

### Не реализовано намеренно

- embeddings категориальных признаков в DNN;
- большие переборы feature engineering;
- подбор ансамбля по Public LB.

Эти идеи исключены, чтобы не усложнять решение без подтверждённого улучшения CV.
