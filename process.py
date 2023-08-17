"""
a plot script for tensorboard download data,
e.g.
---------------------------------------
Wall time,Step,Value
---------------------------------------
1654943296.2150984,0,59.310001373291016
1654943313.7791796,1,64.77999877929688
1654943332.386608,2,70.02999877929688
1654943350.320845,3,71.08000183105469
1654943368.0967004,4,78.98999786376953
1654943386.335942,5,79.69999694824219
...
"""
import os
import sys
from tsmoothie.smoother import *

import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "14"
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
RENAME = {
    "drsom": "SOTRGS (inexact low-rank)",
    "nsgd": "FOTRGS (normalized-SGD)",
    "sgd1": "SGD"
}
LINESTYLE = {
    "drsom": "solid",
    "nsgd": "solid",
    "sgd1": "dashed"
}
MARKER = {
    "drsom": "x"
}
smoother = ExponentialSmoother(window_len=4, alpha=0.05)

dfs = {}
dirin = sys.argv[1]
dirout = sys.argv[2]
for f in os.listdir(dirin):
    if f.endswith('csv'):
        method = "-".join(f.split("-")[2:4]).split("@")[0]
        metric = 'training loss'
        method = method.lower()
        df = pd.read_csv(f"{sys.argv[1]}/{f}")
        df = df.rename(columns={k: k.lower() for k in df.columns})
        if df.shape[0] == 0:
            print(f'no data for {f}')
            continue
        # df['Wall time'] = pd.to_datetime(df['Wall time'])
        df['method'] = method
        df[metric] = df['value']
        df = df.set_index(['method', 'step']).drop(columns=['value'])
        if (method) in dfs:
            dfs[method][metric] = df[metric]
        else:
            dfs[method] = df

fig, ax = plt.subplots(1, 1)
for method in sorted(dfs.keys()):
    vv = smoother.smooth(dfs[method]['training loss'].values).data[0]
    ax.plot(
        np.arange(len(vv)), vv,
        label=RENAME.get(
            method.split("-")[0].lower()
        ) + ("" if method.startswith("drsom") else
             f' ({method.split("-")[1].replace("lr", "").replace("b", "")})'),
        # label=method,  # RENAME.get(method.split("-")[0].lower()),
        linestyle=LINESTYLE.get(method.split("-")[0].lower()),
        marker=MARKER.get(method.split("-")[0].lower()),
        markersize=4,
        alpha=.9,
        linewidth=2.2
    )
ax.set_yscale("log")
fig.legend(
    loc='center right',
    bbox_to_anchor=(0.9, 0.78),
    ncol=1,
    fancybox=True,
    prop={
        'size': 12
    }
)

fig.savefig('fig1.pdf')
