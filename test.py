import numpy as np
from scan import scan_cpd_mean

import kagglehub

path = kagglehub.dataset_download("vishala28/swat-dataset-secure-water-treatment-system")

print("Path to dataset files:", path)

import pandas as pd
df = pd.read_csv('C:/Users/aseelappumud/.cache/kagglehub/datasets/vishala28/swat-dataset-secure-water-treatment-system/versions/3/merged.csv')


x = df["PIT502"].to_numpy()

x_std = (x - np.mean(x)) / np.std(x)

seed = 500
rng = np.random.default_rng(seed)

n = len(np.log(df["PIT502"]))
upper = int(n ** (2/3))

window_sizes = np.sort(rng.choice(np.arange(500, upper + 1), size= 15, replace=False))

cpts, time = scan_cpd_mean(
    series = x_std,
    window_sizes = window_sizes,
    n_perm = 400,
    alpha_q = 1,
    threshold = 0.7,
    workers=48,
    backend="thread",
    batch_size=32,
    seed = 500)

print(len(cpts))

print(time)