import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# ===============================================================
# 1. Data handling
# ===============================================================
def load_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ===============================================================
# 2. Evaluation utilities
# ===============================================================
def evaluate_model(model, X, y, dataset_name, C_value):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    results = []
    for label in sorted(report.keys()):
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            results.append(
                {
                    "C": C_value,
                    "Dataset": dataset_name,
                    "Class": int(label),
                    "Accuracy": acc,
                    "Precision": report[label]["precision"],
                    "F1": report[label]["f1-score"],
                    "Recall": report[label]["recall"],
                }
            )
    return results, acc, y, y_pred


def plot_confusion_matrix(y_true, y_pred, folder, name):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    path = os.path.join(folder, f"{name}_confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix: {path}")
    return cm


# ===============================================================
# 3. Model training and evaluation
# ===============================================================
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, C_range, folder, name):
    all_results = []
    json_output = {}
    train_scores_dict = {C: [] for C in C_range}
    val_scores_dict = {C: [] for C in C_range}
    train_sizes = np.linspace(0.1, 1.0, 10)

    for C in C_range:
        train_f1_list, val_f1_list = [], []

        for frac in train_sizes:
            n_samples = int(len(X_train) * frac)
            X_train_frac = X_train.iloc[:n_samples]
            y_train_frac = y_train.iloc[:n_samples]

            model = SVC(kernel="linear", random_state=77, C=C)
            model.fit(X_train_frac, y_train_frac)

            results_train, _, y_train_true, y_train_pred = evaluate_model(
                model, X_train_frac, y_train_frac, "Train", C
            )
            results_val, _, y_val_true, y_val_pred = evaluate_model(
                model, X_val, y_val, "Validation", C
            )
            results_test, _, y_test_true, y_test_pred = evaluate_model(
                model, X_test, y_test, "Test", C
            )

            all_results.extend(results_train + results_val + results_test)

            json_output[C] = {
                "Train": pd.DataFrame(results_train).to_dict(orient="records"),
                "Validation": pd.DataFrame(results_val).to_dict(orient="records"),
                "Test": pd.DataFrame(results_test).to_dict(orient="records"),
            }

            train_f1_list.append(np.mean([r["F1"] for r in results_train]))
            val_f1_list.append(np.mean([r["F1"] for r in results_val]))

        train_scores_dict[C] = train_f1_list
        val_scores_dict[C] = val_f1_list


        cm = plot_confusion_matrix(y_test_true, y_test_pred, folder, f"{name}_C{C}")

    return all_results, json_output, train_scores_dict, val_scores_dict, train_sizes


# ===============================================================
# 4. Save functions
# ===============================================================
def save_results(folder, all_results, json_output):
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(folder, "svm_detailed_metrics.csv"), index=False)
    with open(os.path.join(folder, "svm_detailed_metrics.json"), "w") as f:
        json.dump(json_output, f, indent=4)
    print(f"✅ Saved results to {folder}")
    return df_results


def save_summary(folder, df_results):
    summary = (
        df_results.groupby(["C", "Dataset"])
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    with open(os.path.join(folder, "svm_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


# ===============================================================
# 5. Learning curve visualization
# ===============================================================
def plot_learning_curves(name, folder, train_sizes, train_scores_dict, val_scores_dict, C_range):

    avg_val_f1 = {C: np.mean(val_scores_dict[C]) for C in C_range}
    best_C = max(avg_val_f1, key=avg_val_f1.get)
    print(f"🌟 Best C based on validation F1: {best_C}")


    plt.figure(figsize=(8, 6))
    plt.plot(C_range, list(avg_val_f1.values()), "o-", color="tab:blue")
    plt.xscale("log")
    plt.xlabel("C value (log scale)")
    plt.ylabel("Average Validation F1-score")
    plt.title(f"F1 vs C - {name}")
    for c, v in avg_val_f1.items():
        plt.text(c, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "svm_F1_vs_C.png"), dpi=300)
    plt.close()


# ===============================================================
# 6. Fairness analysis
# ===============================================================
def fairness_analysis(group1_name, group2_name, parent_folder="fairness"):

    folder = os.path.join(parent_folder, f"{group1_name}_vs_{group2_name}")
    os.makedirs(folder, exist_ok=True)


    cm1 = pd.read_csv(os.path.join(group1_name, "svm_detailed_metrics.csv"))
    cm2 = pd.read_csv(os.path.join(group2_name, "svm_detailed_metrics.csv"))


    g1_metrics = cm1.groupby("Dataset").mean(numeric_only=True)
    g2_metrics = cm2.groupby("Dataset").mean(numeric_only=True)


    gap = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Gap": [
            g1_metrics["Accuracy"].mean() - g2_metrics["Accuracy"].mean(),
            g1_metrics["Precision"].mean() - g2_metrics["Precision"].mean(),
            g1_metrics["F1"].mean() - g2_metrics["F1"].mean(),
            g1_metrics["F1"].mean() - g2_metrics["F1"].mean(),
        ],
    })


    g1_metrics.to_csv(os.path.join(folder, f"{group1_name}_metrics.csv"))
    g2_metrics.to_csv(os.path.join(folder, f"{group2_name}_metrics.csv"))
    gap.to_csv(os.path.join(folder, "fairness_gap.csv"), index=False)


    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    plt.bar(x - 0.15, [g1_metrics[m].mean() for m in metrics], 0.3, label=group1_name)
    plt.bar(x + 0.15, [g2_metrics[m].mean() for m in metrics], 0.3, label=group2_name)
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title(f"Group Comparison - {group1_name} vs {group2_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "group_bar_chart.png"), dpi=300)
    plt.close()


    plt.bar(["ΔTPR (≈Recall)", "ΔFPR (≈1-Precision)"], [gap.iloc[2, 1], -gap.iloc[1, 1]])
    plt.ylabel("Difference")
    plt.title(f"ΔTPR / ΔFPR - {group1_name} vs {group2_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "deltaTPR_deltaFPR.png"), dpi=300)
    plt.close()

    print(f"✅ Saved fairness analysis to {folder}")


# ===============================================================
# 7. Main pipeline
# ===============================================================
def main(df, name):
    folder = os.path.join(os.getcwd(), name)
    os.makedirs(folder, exist_ok=True)
    X, y = load_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    C_range = [0.001, 0.01, 0.1, 1, 10]

    all_results, json_output, train_scores_dict, val_scores_dict, train_sizes = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, C_range, folder, name
    )

    df_results = save_results(folder, all_results, json_output)
    save_summary(folder, df_results)
    plot_learning_curves(name, folder, train_sizes, train_scores_dict, val_scores_dict, C_range)


if __name__ == "__main__":
    dataset_names = [
        "female_subset",
        "male_subset",
        "highedu_subset",
        "lowedu_subset",
        "rural_subset_modified",
        "urban_subset_modified",
        "A1_low_cost_topk",
        "All_feature",
        "All_feature_with_interactions",
    ]

    for name in dataset_names:
        df = pd.read_csv(f"{name}.csv")
        main(df, name)


    fairness_analysis("female_subset", "male_subset", "gender")
    fairness_analysis("rural_subset_modified", "urban_subset_modified", "region")
    fairness_analysis("highedu_subset", "lowedu_subset", "education")