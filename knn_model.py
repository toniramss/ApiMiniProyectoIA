# knn_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ====== CARGA Y PREPROCESADO (una sola vez) ======
df = pd.read_csv("data/wine_quality_merged.csv")

df["quality_bin"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

X = df.drop(["quality", "quality_bin"], axis=1)
y = df["quality_bin"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ====== FUNCIÓN DEL MODELO ======
def run_knn(k: int):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "k": k,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 2),
        "precision": round(float(precision_score(y_test, y_pred)), 2),
        "recall": round(float(recall_score(y_test, y_pred)), 2),
        "f1": round(float(f1_score(y_test, y_pred)), 2),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def knn_curve(min_k: int = 1, max_k: int = 40):
    points = []

    for k in range(min_k, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        points.append({
            "k": k,
            "accuracy": round(float(acc), 4)
        })

    return points