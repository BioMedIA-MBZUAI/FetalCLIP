import torch
import numpy as np
import matplotlib.pyplot as plt


PATH_DOT_PRODS = 'FetalCLIP_dot_prods_map.pt'
PATH_SAVE_PLOT = 'FetalCLIP_ga_estimation.png'

TOP_N_PROBS = 15

# REF https://srhr.org/fetalgrowthcalculator/#/
DATA_MIN_HC = 100 # Corresponding to GA 14 weeks (50th percentile)
DATA_MAX_HC = 342 # Corresponding to GA 40 weeks (50th percentile)

# DATA_MIN_HC = 138 # 2.5th --> GA pretraining
# DATA_MAX_HC = 276 # 97.5th --> GA pretraining

# DATA_MIN_HC = 143 # 5th --> GA pretraining
# DATA_MAX_HC = 256 # 95th --> GA pretraining

# DATA_MIN_HC = 157 # 25th --> GA pretraining
# DATA_MAX_HC = 212 # 75th --> GA pretraining

list_ga_in_days = [weeks * 7 + days for weeks in range(14, 38) for days in range(0, 7)]

exp_data = torch.load(PATH_DOT_PRODS, map_location='cpu')

list_true_hc = [d['true_hc'] for d in exp_data]
list_dot_prods = [d['text_dot_prods'] for d in exp_data]

def get_hc_from_days(t, quartile='0.5'):
    t = t / 7

    dict_params = {
        '0.025': [1.59317517131532e+0, 2.9459800552433e-1,  -7.3860372566707e-3,  6.56951770216148e-5, 0e+0],
        '0.5'  : [2.09924879247164e+0, 2.53373656106037e-1, -6.05647816678282e-3, 5.14256072059917e-5, 0e+0],
        '0.975': [2.50074069629423e+0, 2.20067854715719e-1, -4.93623111462443e-3, 3.89066000946519e-5, 0e+0],
    }

    b0, b1, b2, b3, b4 = dict_params[quartile]

    hc_q50 = np.exp(
        b0 + b1*t + b2*t**2 + b3*t**3 + b4*t**4
    )

    return hc_q50

def find_median_from_top_n(text_dot_prods, n=20):
    assert len(text_dot_prods.shape) == 1
    tmp = [[i, t] for i, t in enumerate(text_dot_prods)]
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)[:n]
    tmp = sorted(tmp, key=lambda x: x[0])
    median_ind = tmp[n // 2][0]
    return median_ind

list_hc_pred = []
list_pred_days = []
for text_dot_prods, true_hc in zip(list_dot_prods, list_true_hc):
    med_indices = find_median_from_top_n(text_dot_prods[0], TOP_N_PROBS)
    med_values = text_dot_prods[0][med_indices]
    pred_days = list_ga_in_days[med_indices]
    pred_hc = get_hc_from_days(pred_days)
    list_hc_pred.append(pred_hc)
    list_pred_days.append(pred_days)

list_pred_days = [d for hc, d in zip(list_true_hc, list_pred_days) if DATA_MIN_HC <= hc <= DATA_MAX_HC]
list_true_hc   = [hc for hc in list_true_hc if DATA_MIN_HC <= hc <= DATA_MAX_HC]

true = np.array(list_true_hc)
pred = np.array(list_pred_days)

q_low  = [get_hc_from_days(ga, '0.025') for ga in list_ga_in_days]
q_mid  = [get_hc_from_days(ga, '0.5')   for ga in list_ga_in_days]
q_high = [get_hc_from_days(ga, '0.975') for ga in list_ga_in_days]

list_validity = []
for t, p in zip(true, pred):
    _q_low = get_hc_from_days(p, '0.025')
    _q_high = get_hc_from_days(p, '0.975')
    if _q_low <= t <= _q_high:
        list_validity.append(1)
    else:
        list_validity.append(0)

len(list_hc_pred), len(list_pred_days)

valid_true = [t for t, v in zip(true, list_validity) if v == 1]
valid_pred = [p for p, v in zip(pred, list_validity) if v == 1]
invalid_true = [t for t, v in zip(true, list_validity) if v == 0]
invalid_pred = [p for p, v in zip(pred, list_validity) if v == 0]

# Compute statistics
total_predictions = len(list_validity)
valid_count = sum(list_validity)
valid_percentage = (valid_count / total_predictions) * 100 if total_predictions > 0 else 0

# Computer 95% CI using bootstrap
np_list_validity = np.array(list_validity)

B = 10000  # number of bootstrap resamples
boot_means = np.empty(B)

rng = np.random.default_rng(seed=42) # reproducible

for b in range(B):
    sample = rng.choice(np_list_validity, size=len(np_list_validity), replace=True)
    boot_means[b] = sample.mean()

ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

# Print results
print(f"Total Predictions: {total_predictions}")
print(f"Valid Predictions: {valid_count}")
print(f"Valid Prediction Rate: {valid_percentage:.2f}%")
print(f"Point estimate: {np_list_validity.mean()*100:.2f}%")
print(f"95% bootstrap CI:, ({ci_lower*100:.2f}, {ci_upper*100:.2f})")

plt.style.use("seaborn-v0_8-paper")

plt.plot(q_mid, list_ga_in_days, color="black", linestyle="-", linewidth=2)#, label="Ref. curve P50")
plt.plot(q_low, list_ga_in_days, color="darkorange", linestyle="--", linewidth=2)#, label="Ref. curve P2.5")
plt.plot(q_high, list_ga_in_days, color="darkorange", linestyle="--", linewidth=2)#, label="Ref. curve P97.5")
plt.scatter(valid_true, valid_pred, color="royalblue", s=30, edgecolor='k', label='Valid prediction')
plt.scatter(invalid_true, invalid_pred, color="red", s=30, edgecolor='k', label='Invalid prediction')

plt.xlabel("True HC (mm)", fontsize=12)
plt.ylabel("Predicted GA (days)", fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(False)

handles, labels = plt.gca().get_legend_handles_labels()
new_handles = [plt.Line2D([], [], marker='o', color=h.get_facecolor()[0], markersize=10, linestyle='') for h in handles]

plt.legend(handles=new_handles, labels=labels, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(PATH_SAVE_PLOT, dpi=400, format='png', bbox_inches='tight')

plt.show()