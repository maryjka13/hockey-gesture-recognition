import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import classification_report


df = pd.read_csv("features_5.csv")

X = df.drop(columns=["label", "gesture", "video"])
y = df["label"]

feature_names = X.columns.tolist()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
best_model = None
best_score = 0


for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    scores.append(acc)

    print(f"Fold {fold} accuracy: {acc:.3f}")

    if acc > best_score:
        best_score = acc
        best_model = model



print("\nCV accuracy mean:", np.mean(scores))
print("CV accuracy std :", np.std(scores))


# feature importance
importance = pd.Series(best_model.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=False)

print("\nTop 20 features:")
print(importance.head(20))

print(classification_report(y_test, preds))


joblib.dump(best_model, "xgb_gesture_model_5.pkl")
joblib.dump(scaler, "scaler_5.pkl")
joblib.dump(feature_names, "feature_names_5.pkl")

print("\nModel saved!")
