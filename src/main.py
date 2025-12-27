from pathlib import Path
import pandas as pd
from predictor import train_model, predict_risk


def main():
    print("🚀 Student Performance Predictor")
    print("Загрузка данных...")

    # Получаем путь к текущему файлу main.py
    current_file_path = Path(__file__).resolve()
    # Строим путь к файлу sample.csv
    data_file = current_file_path.parent.parent / 'data' / 'sample.csv'

    print(f"Ищу файл: {data_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"Файл не найден: {data_file}")

    df = pd.read_csv(data_file)
    print(f"Загружено {len(df)} студентов.")

    # Обучаем модель
    print("Обучение модели...")
    model = train_model(df)
    print("Модель обучена!")

    # Пример предсказания
    student = {
        'grade1': 40, 'grade2': 50, 'grade3': 45,
        'attended_lectures': 6, 'total_lectures': 20,
        'late_assignments': 3, 'total_assignments': 4
    }

    is_at_risk, probability = predict_risk(model, student)

    print("\nСтудент:")
    for key, value in student.items():
        print(f"  {key}: {value}")

    print(f"Риск отчисления: {probability:.2%}")
    risk_status = "находится" if is_at_risk else "не находится"
    print(f"Студент {risk_status} в группе риска.")


if __name__ == "__main__":
    main()
