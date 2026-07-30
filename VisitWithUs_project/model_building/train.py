
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import mlflow

#Load the train and test files
X_train = pd.read_csv('VisitWithUs_project/data/X_train.csv')
X_test = pd.read_csv('VisitWithUs_project/data/X_test.csv')
y_train = pd.read_csv('VisitWithUs_project/data/y_train.csv')
y_test  = pd.read_csv('VisitWithUs_project/data/y_test.csv')

# Define categorical and numerical features
categorical_features = X_train.select_dtypes(include='object').columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create a column transformer for preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Create the full pipeline including preprocessor and XGBoost classifier
model_pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
)

# Define hyperparameters for GridSearchCV
param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__max_depth': [3, 5, 7]
}

# Perform GridSearchCV to find the best model parameters
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

#Log the parameters and metrics to MLflow for experimentation tracking
mlflow.set_experiment("Wellness_Tourism_Package_Prediction")
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

    # Evaluate on test set again for consistent logging
    y_pred_test = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    mlflow.log_metric("test_accuracy", report['accuracy'])
    mlflow.log_metric("test_precision_class_0", report['0']['precision'])
    mlflow.log_metric("test_recall_class_0", report['0']['recall'])
    mlflow.log_metric("test_f1_class_0", report['0']['f1-score'])
    mlflow.log_metric("test_precision_class_1", report['1']['precision'])
    mlflow.log_metric("test_recall_class_1", report['1']['recall'])
    mlflow.log_metric("test_f1_class_1", report['1']['f1-score'])
    mlflow.log_metric("test_macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("test_macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("test_macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("test_weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("test_weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("test_weighted_avg_f1", report['weighted avg']['f1-score'])

    # Log the model with the 'name' parameter and an 'input_example'
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        registered_model_name="XGBoostCustomerPurchasePredictor",
        input_example=X_train.iloc[[0]]
    )

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, 'VisitWithUs_project/deployment/best_model.pkl')
print("Trained model saved as 'VisitWithUs_project/deployment/best_model.pkl'")
