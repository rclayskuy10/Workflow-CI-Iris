"""
Modelling untuk MLflow Project - Covertype Classification
Nama: Riski Pratama
File ini dijalankan melalui MLflow Project (mlflow run).
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    # Load data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(data_dir, 'covtype_train_preprocessed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'covtype_test_preprocessed.csv'))

    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Enable autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1)
        mlflow.log_param("model_type", "RandomForestClassifier")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
