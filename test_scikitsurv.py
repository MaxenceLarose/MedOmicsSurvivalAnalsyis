import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

# Data
data_x, data_y = load_veterans_lung_cancer()
data_x_numeric = OneHotEncoder().fit_transform(data_x)

print(data_x_numeric)
print(data_y)


# Fit
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, data_y)

# New dataset
x_new = pd.DataFrame.from_dict(
    {
        1: [65, 0, 0, 1, 60, 1, 0, 1],
        2: [65, 0, 0, 1, 60, 1, 0, 0],
        3: [65, 0, 1, 0, 60, 1, 0, 0],
        4: [65, 0, 1, 0, 60, 1, 0, 1],
    },
    columns=data_x_numeric.columns,
    orient="index",
)

# Holdout pred
pred_surv = estimator.predict_survival_function(x_new)

print(pred_surv)

# Holdout plot
time_points = np.arange(1, 1000)
for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()
