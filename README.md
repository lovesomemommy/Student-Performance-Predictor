# Student Performance Predictor 🎓

Предсказывает, находится ли студент в группе риска отчисления на основе его академической активности.

## 📌 Установка

```bash
git clone https://github.com/lovesomemommy/Student-Performance-Predictor.git
cd Student-Performance-Predictor
pip install -r requirements.txt

```

## ▶️ Пример использования

```python
from src.predictor import train_model, predict_risk
import pandas as pd

df = pd.read_csv("data/sample.csv")
model = train_model(df)

student = {
    "grade1": 40, "grade2": 50, "grade3": 45,
    "attended_lectures": 6, "total_lectures": 20,
    "late_assignments": 3, "total_assignments": 4
}

at_risk, probability = predict_risk(model, student)
print(f"Риск отчисления: {probability:.2%}")

```

## 📂 Структура проекта

- `src/` — основной код
- `tests/` — unit-тесты
- `data/` — примеры данных
- `docs/` — документация

## 🧪 Тестирование
Запуск тестов:

```bash
pytest
```

С покрытием кода:

```bash
pytest --cov=src tests/
```

## 📦 Требования

- Python 3.8+
- Библиотеки из `requirements.txt`:
  - pandas >=1.3.0
  - scikit-learn >=1.0.0
  - numpy >=1.20.0
  - pytest, flake8, black

## 🚀 CI/CD

[![CI](https://github.com/lovesomemommy/Student-Performance-Predictor/workflows/CI/badge.svg)](https://github.com/lovesomemommy/Student-Performance-Predictor/actions)

Еженедельный отчёт доступен на [GitHub Pages](https://lovesomemommy.github.io/Student-Performance-Predictor/report.html).

## 📄 Лицензия
Этот проект распространяется под лицензией MIT.

## 👩‍💻 Автор
lovesomemommy