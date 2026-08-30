import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# The GitHub Actions workflow starts an MLflow server on localhost:5000.
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visit-with-us-wellness-purchase-experiment")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze().astype(int)
ytest = pd.read_csv("ytest.csv").squeeze().astype(int)

numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print("Scale positive class weight:", class_weight)

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.10],
}

classification_threshold = 0.45

with mlflow.start_run():
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every hyperparameter combination as a nested MLflow run.
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric(
                "mean_test_recall",
                float(results["mean_test_score"][i]),
            )
            mlflow.log_metric(
                "std_test_recall",
                float(results["std_test_score"][i]),
            )

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True, zero_division=0
    )

    print("Best parameters:")
    print(grid_search.best_params_)
    print("\nTest classification report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
        "classification_threshold": classification_threshold,
    })

    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = (
        "tourism_project/deployment/"
        "best_wellness_tourism_model_v1.joblib"
    )

    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"\nBest model saved to: {model_path}")
