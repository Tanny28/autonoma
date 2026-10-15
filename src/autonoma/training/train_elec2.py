import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

MODEL_NAME = "AutonomaElec2"
EXPERIMENT_NAME = "autonoma-elec2-baseline"
TRAIN_FRACTION = 0.30
RANDOM_STATE = 42


def load_elec2():
    dataset = fetch_openml(data_id=151, as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.copy()
    return features, target


def build_model() -> Pipeline:
    categorical_features = ["day"]
    numeric_features = [
        "date",
        "period",
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    features, target = load_elec2()

    # Preserve the original temporal order.
    split_index = int(len(features) * TRAIN_FRACTION)

    features_train = features.iloc[:split_index].copy()
    features_test = features.iloc[split_index:].copy()

    target_train_raw = target.iloc[:split_index].copy()
    target_test_raw = target.iloc[split_index:].copy()

    label_encoder = LabelEncoder()
    target_train = label_encoder.fit_transform(target_train_raw)
    target_test = label_encoder.transform(target_test_raw)

    model = build_model()

    with mlflow.start_run(run_name="elec2-baseline") as run:
        model.fit(features_train, target_train)

        predictions = model.predict(features_test)

        accuracy = accuracy_score(target_test, predictions)
        balanced_accuracy = balanced_accuracy_score(target_test, predictions)

        mlflow.log_param("dataset", "Elec2")
        mlflow.log_param("dataset_rows", len(features))
        mlflow.log_param("feature_count", features.shape[1])
        mlflow.log_param("train_fraction", TRAIN_FRACTION)
        mlflow.log_param("train_rows", len(features_train))
        mlflow.log_param("evaluation_rows", len(features_test))
        mlflow.log_param("classifier", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param(
            "label_mapping",
            dict(
                zip(
                    label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_),
                    strict=True,
                )
            ),
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("balanced_accuracy", balanced_accuracy)

        signature = infer_signature(
            features_train,
            model.predict(features_train),
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    matching_versions = [version for version in versions if version.run_id == run_id]

    if not matching_versions:
        raise RuntimeError(f"Could not find registered model version for run {run_id}")

    model_version = max(
        matching_versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        model_version.version,
    )

    print()
    print("Elec2 baseline training complete.")
    print(f"Rows: {len(features)}")
    print(f"Training rows: {len(features_train)}")
    print(f"Evaluation rows: {len(features_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"MLflow run: {run_id}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Registered version: {model_version.version}")
    print(f"Alias: {MODEL_NAME}@champion")


if __name__ == "__main__":
    main()
