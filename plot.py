import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


epsoids = 20
col_name = "Evaluation/AverageDiscountedReturn"
path1 = "/nfsshare/home/lichenxi/DRO/DRO_trustregion-correct/sgd.csv"
path2 = "/nfsshare/home/lichenxi/DRO/drsom.csv"


sgd = pd.read_csv(path1)
drsom = pd.read_csv(path2)


plt.figure(dpi=300, figsize=(8, 4))
plt.title("")
plt.plot(np.arange(1, epsoids + 1), sgd["all_avg_mae"], label="sgd")
plt.plot(np.arange(1, 16), drsom["all_avg_mae"], label="drsom")
# plt.plot(np.arange(1, epsoids + 1), vpg_2[col_name], label='pure pg, a=1e-1')
# plt.plot(np.arange(1, epsoids + 1), vpg_adam[col_name], label='adam')
# plt.plot(np.arange(1, epsoids + 1), vpg_sgd[col_name], label='sgd')
# plt.plot(np.arange(1, epsoids + 1), vpg_adagrad[col_name], label='adagrad')
# plt.plot(np.arange(1, epsoids + 1), vpg_drsom_no_hessian[col_name], label='drsom without hessian')
# plt.plot(np.arange(1, epsoids + 1), vpg_drsom[col_name], label='drsom')


plt.legend()
plt.ylabel("MAE")
plt.xlabel("Epoch #")
plt.savefig("309.jpg")
plt.show()
