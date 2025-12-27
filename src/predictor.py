import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def prepare_features(df):
    """Подготавливает признаки для модели."""
    df = df.copy()
    df["avg_grade"] = df[["grade1", "grade2", "grade3"]].mean(axis=1)
    df["attendance_rate"] = df["attended_lectures"] / df["total_lectures"]
    df["late_ratio"] = df["late_assignments"] / df["total_assignments"]
    return df[["avg_grade", "attendance_rate", "late_ratio"]]


def train_model(df, target_col="at_risk"):
    """Обучает модель предсказания риска отчисления."""
    X = prepare_features(df)
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def predict_risk(model, student_data):
    """Предсказывает риск отчисления для одного студента."""
    student_df = pd.DataFrame([student_data])
    X = prepare_features(student_df)
    prob = model.predict_proba(X)[0][1]  # вероятность класса "at_risk = 1"
    is_at_risk = bool(prob > 0.5)
    return is_at_risk, prob
