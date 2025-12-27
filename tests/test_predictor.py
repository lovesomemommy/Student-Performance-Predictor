from pathlib import Path
import pandas as pd
from src.predictor import prepare_features, train_model, predict_risk


def test_prepare_features():
    df = pd.DataFrame(
        {
            "grade1": [80],
            "grade2": [70],
            "grade3": [75],
            "attended_lectures": [10],
            "total_lectures": [20],
            "late_assignments": [2],
            "total_assignments": [5],
            "at_risk": [0],
        }
    )
    X = prepare_features(df)
    assert X["avg_grade"].iloc[0] == 75.0
    assert X["attendance_rate"].iloc[0] == 0.5
    assert X["late_ratio"].iloc[0] == 0.4


def _load_sample_data():
    current_file_path = Path(__file__).resolve()
    data_file = current_file_path.parent.parent / "data" / "sample.csv"
    print(f"Ищу файл: {data_file}")
    print(f"Файл существует? {data_file.exists()}")
    if not data_file.exists():
        raise FileNotFoundError(f"Файл не найден: {data_file}")
    return pd.read_csv(data_file)


def test_train_model():
    df = _load_sample_data()
    model = train_model(df)
    assert hasattr(model, "predict")


def test_predict_risk():
    df = _load_sample_data()
    model = train_model(df)
    student = {
        "grade1": 40,
        "grade2": 45,
        "grade3": 50,
        "attended_lectures": 5,
        "total_lectures": 20,
        "late_assignments": 4,
        "total_assignments": 5,
    }
    is_at_risk, prob = predict_risk(model, student)
    assert isinstance(is_at_risk, bool)
    assert 0.0 <= prob <= 1.0


def test_low_risk_student():
    df = _load_sample_data()
    model = train_model(df)
    student = {
        "grade1": 90,
        "grade2": 92,
        "grade3": 88,
        "attended_lectures": 19,
        "total_lectures": 20,
        "late_assignments": 0,
        "total_assignments": 5,
    }
    is_at_risk, prob = predict_risk(model, student)
    assert not is_at_risk


def test_high_risk_student():
    df = _load_sample_data()
    model = train_model(df)
    student = {
        "grade1": 30,
        "grade2": 35,
        "grade3": 40,
        "attended_lectures": 4,
        "total_lectures": 20,
        "late_assignments": 5,
        "total_assignments": 5,
    }
    is_at_risk, prob = predict_risk(model, student)
    assert is_at_risk
