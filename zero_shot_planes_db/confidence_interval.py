import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tqdm import tqdm

# --------------------------------------------------
# 1) Load JSON files
# --------------------------------------------------
with open("test_five_planes_prediction.json", "r") as f:
    data_planes = json.load(f)

with open("test_brain_subplanes_prediction.json", "r") as f:
    data_brain = json.load(f)

# --------------------------------------------------
# 2) Merge predictions & labels
# --------------------------------------------------
y_pred = np.array(data_planes["prediction"] + data_brain["prediction"])
y_true = np.array(data_planes["label"] + data_brain["label"])

labels = sorted(set(y_true))

# --------------------------------------------------
# 3) Bootstrap
# --------------------------------------------------
n_bootstrap = 10_000
rng = np.random.default_rng(seed=42)

boot_f1_per_class = []
boot_f1_mean = []

n = len(y_true)

for _ in tqdm(range(n_bootstrap)):
    idx = rng.integers(0, n, n)
    yt = y_true[idx]
    yp = y_pred[idx]

    f1_pc = f1_score(
        yt, yp,
        labels=labels,
        average=None,
        zero_division=0
    )

    boot_f1_per_class.append(f1_pc)
    boot_f1_mean.append(f1_pc.mean())

boot_f1_per_class = np.array(boot_f1_per_class)
boot_f1_mean = np.array(boot_f1_mean)

# --------------------------------------------------
# 4) Confidence Intervals
# --------------------------------------------------
ci_low = 2.5
ci_high = 97.5

rows = []
for i, cls in enumerate(labels):
    rows.append({
        "class": cls,
        "f1_mean": boot_f1_per_class[:, i].mean(),
        "ci_low": np.percentile(boot_f1_per_class[:, i], ci_low),
        "ci_high": np.percentile(boot_f1_per_class[:, i], ci_high),
        "dci": (np.percentile(boot_f1_per_class[:, i], ci_high) -  np.percentile(boot_f1_per_class[:, i], ci_low)) / 2,
    })

df_f1_ci = pd.DataFrame(rows)

mean_f1_ci = {
    "mean_f1": boot_f1_mean.mean(),
    "ci_low": np.percentile(boot_f1_mean, ci_low),
    "ci_high": np.percentile(boot_f1_mean, ci_high),
}

print(df_f1_ci)
print("\nMean F1 CI:", mean_f1_ci)
