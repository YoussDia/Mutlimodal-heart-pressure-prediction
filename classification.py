"""
Modèle de classification ECG/SCG — Split par patient + LOO
═══════════════════════════════════════════════════════════
Contexte :
  - 42 patients (2 en classe 1, 9 en classe 2, 28 en classe 3, 3 en classe 4)
  - Split STRICT par patient : aucun patient ne chevauche train et test
  - Agrégation au niveau patient (mean + std + median des battements)
  - Leave-One-Out CV : seule évaluation fiable avec si peu de patients
  - 4 modèles comparés, meilleur sélectionné automatiquement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#changement des donees

df = pd.read_csv("ecg_scg_segment_dataset.csv")
print(f"Shape original         : {df.shape}")
print(f"Patients               : {df['patient_id'].nunique()}")
print(f"Segments               : {df['segment_name'].nunique()}")

#netoyage des données

# supression des lignes vide et du premier battement
df = df.dropna(subset=["classe"])
df = df[df["beat_index"] != 0]

# remplacement de l'intervalle  ivrt_ms (nulle) par la médiane de ce même intervalle pour les segments du même patient
df.loc[df["ivrt_ms"].isna(), "ivrt_ms"] = df.loc[df["ivrt_ms"].isna(), "median_ivrt_ms"]
#supression des lignes vides restantes
df = df.dropna()

print(f"Shape après nettoyage  : {df.shape}")
print(f"NaN restants           : {df.isnull().sum().sum()}")

#AGRÉGATION AU NIVEAU PATIENT
#pour chaque patient, une ligne correspond à la moyenne, l'écart-type et la médiane de chaque feature sur tous les segments du patient
BEAT_FEATS = [
    "pr_ms", "qt_ms", "rr_s", "hr_bpm",
    "pep_ms", "et_ms", "ivct_ms", "ivrt_ms",
]
SEG_FEATS = [
    "mean_pr_ms", "median_pr_ms", "mean_qt_ms", "median_qt_ms",
    "mean_rr_s",  "median_rr_s",  "mean_hr_bpm", "median_hr_bpm",
    "mean_pep_ms","median_pep_ms","mean_et_ms",  "median_et_ms",
    "mean_ivct_ms","median_ivct_ms","mean_ivrt_ms","median_ivrt_ms",
]

agg_dict = {}
for f in BEAT_FEATS:
    agg_dict[f"{f}_mean"]   = (f, "mean")
    agg_dict[f"{f}_std"]    = (f, "std")
    agg_dict[f"{f}_median"] = (f, "median")
for f in SEG_FEATS:
    agg_dict[f] = (f, "mean")

patient_df = (
    df.groupby(["patient_id", "classe"])
    .agg(**agg_dict)
    .reset_index()
    .fillna(0)  # std=0 si un seul segment
)

FEAT_COLS = [c for c in patient_df.columns if c not in ["patient_id", "classe"]]
X        = patient_df[FEAT_COLS].values
y        = patient_df["classe"].astype(int).values
patients = patient_df["patient_id"].values

print(f"\nDataset patient-level  : {len(y)} patients × {len(FEAT_COLS)} features")
print("\nDistribution des classes :")
for cls, cnt in zip(*np.unique(y, return_counts=True)):
    bar = "█" * cnt
    print(f"  Classe {cls} : {cnt:2d} patients  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
#    MODÈLES + LEAVE-ONE-OUT CROSS-VALIDATION
#    LOO = Chaque fold : train sur tous les patients sauf 1, test sur ce patient seul
# ══════════════════════════════════════════════════════════════════════════════
loo = LeaveOneOut()

models = {
    "Random Forest" : RandomForestClassifier(
        n_estimators=500, max_depth=5,
        class_weight="balanced", random_state=42
    ),
    "Gradient Boost" : GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, random_state=42
    ),
    "SVM (RBF)" : Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)),
    ]),
    "Logistic Reg." : Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            C=0.5, max_iter=1000,
            class_weight="balanced",

            random_state=42
        )),
    ]),
}

print(f"\n{'═'*55}")
print(f"  ÉVALUATION LOO — {len(y)} patients, {len(y)} folds")
print(f"{'═'*55}")

results     = {}
best_name   = None
best_acc    = 0.0
best_preds  = None

for name, model in models.items():
    preds = cross_val_predict(model, X, y, cv=loo)
    acc   = (preds == y).mean()
    results[name] = {"acc": acc, "preds": preds}
    flag = "  ← meilleur" if acc > best_acc else ""
    print(f"  {name:<18} : {acc:.4f}  ({(preds==y).sum()}/{len(y)} corrects){flag}")
    if acc > best_acc:
        best_acc, best_name, best_preds = acc, name, preds

print(f"{'═'*55}\n")

# ══════════════════════════════════════════════════════════════════════════════
# meilleur modèle : classification_report + matrice de confusion + prédictions par patient
# ══════════════════════════════════════════════════════════════════════════════
print(f"Meilleur modèle : {best_name}  (accuracy LOO = {best_acc:.4f})\n")
print(classification_report(
    y, best_preds,
    target_names=["Classe 1", "Classe 2", "Classe 3", "Classe 4"],
    zero_division=0
))

print(f"\n{'Patient':<30} {'Réel':>6} {'Prédit':>7}  {'':>4}")
print("─" * 52)
for pat, true, pred in zip(patients, y, best_preds):
    status = "1" if true == pred else "0"
    print(f"{pat:<30} {true:>6} {pred:>7}   {status}")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# ── Comparaison des modèles 
ax0 = fig.add_subplot(gs[0, 0])
names_list = list(results.keys())
accs_list  = [results[n]["acc"] for n in names_list]
colors     = ["#2ecc71" if n == best_name else "#bdc3c7" for n in names_list]
bars = ax0.bar(names_list, accs_list, color=colors, edgecolor="white", linewidth=1.5)
ax0.axhline(1/4, color="red", linestyle="--", alpha=0.6, label="Hasard (25%)")
ax0.axhline(1/3, color="orange", linestyle="--", alpha=0.6, label="Baselines (33%)")
for bar, acc in zip(bars, accs_list):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")
ax0.set_ylim(0, 1); ax0.set_title("Comparaison des modèles (LOO)", fontsize=12)
ax0.set_ylabel("Accuracy"); ax0.legend(fontsize=9); ax0.grid(axis="y", alpha=0.3)
ax0.tick_params(axis="x", rotation=15)

# ── Matrice de confusion meilleur modèle 
ax1 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y, best_preds, labels=[1, 2, 3, 4])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["C1","C2","C3","C4"],
            yticklabels=["C1","C2","C3","C4"])
ax1.set_title(f"Confusion — {best_name}\n(LOO, niveau patient)", fontsize=12)
ax1.set_xlabel("Prédit"); ax1.set_ylabel("Réel")

# ── Distribution des classes 
ax2 = fig.add_subplot(gs[0, 2])
cls_labels = [f"Classe {c}" for c in np.unique(y)]
cls_counts = [np.sum(y == c) for c in np.unique(y)]
ax2.bar(cls_labels, cls_counts, color=["#e74c3c","#3498db","#2ecc71","#f39c12"])
for i, (lbl, cnt) in enumerate(zip(cls_labels, cls_counts)):
    ax2.text(i, cnt + 0.1, str(cnt), ha="center", fontweight="bold")
ax2.set_title("Distribution des patients par classe", fontsize=12)
ax2.set_ylabel("Nombre de patients"); ax2.grid(axis="y", alpha=0.3)


plt.savefig("resultats_classification_patient_loo.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGraphiques sauvegardés : resultats_classification_patient_loo.png")