import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


confidence_log = os.path.join("logs-debug", "hb-ddpg", "AntBulletEnv-v0_2", "dynamicskopie.npz")
data = np.load(confidence_log, allow_pickle=True)

gt = data["ground_truth"]
pred_mean = data["predictions_mean"]
pred_std = data["predictions_std"]
pred_median = data["predictions_median"]
pred_q1 = data["predictions_q1"]
pred_q3 = data["predictions_q3"]


x = data["x"].flatten()
X = [n for n in range(len(x))]

#t = []
#for n in range(len(pred_q1)):
#    t.append(np.max(pred_q1[n]))
#t = np.array(t)
#c = np.max(pred_q1, axis=(1, 2))
#q1_max = np.max(np.max(pred_q1, axis=2), axis=1)
#q3_max = np.min(np.min(pred_q3, axis=2), axis=1)
q1_max = np.max(pred_q1, axis=(1, 2))
q3_max = np.max(pred_q3, axis=(1, 2))
iqr_max = q3_max - q1_max

q1_min = np.min(pred_q1, axis=(1, 2))
q3_min = np.min(pred_q3, axis=(1, 2))
iqr_min = q3_min - q1_min

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
axs.plot(X, q1_max, label="q1_max")
axs.plot(X, q3_max, label="q3_max")
axs.plot(X, iqr_max, label="iqr_max")
axs.plot(X, q1_min, label="q1_min")
axs.plot(X, q3_min, label="q3_min")
axs.plot(X, iqr_min, label="iqr_min")

axs.legend()

dimension = 0
pred_mean = gt - pred_mean
mu = [pred_mean[n, :, dimension] for n in range(len(x))]
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
axs.boxplot(mu)
axs.legend()

plt.show()

exit()





dimension = 0

mu = [pred_mean[n, :, dimension] for n in range(len(x))]
err = [np.abs(gt[n, :, dimension] - pred_mean[n, :, dimension]) for n in range(len(x))]
ystd = np.max(np.quantile(pred_mean, axis=1, q=0.25), axis=1)
y1 = np.max(np.quantile(pred_mean, axis=1, q=0.5), axis=1)
y3 = np.max(np.quantile(pred_mean, axis=1, q=0.75), axis=1)


fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
axs.boxplot(mu)
axs.plot(X, ystd)
axs.plot(X, y1)
axs.plot(X, y3)

ystd = np.quantile(pred_mean, axis=1, q=0.25)[:, dimension]
y1 = np.quantile(pred_mean, axis=1, q=0.5)[:, dimension]
y3 = np.quantile(pred_mean, axis=1, q=0.75)[:, dimension]

axs.plot(X, ystd)
axs.plot(X, y1)
axs.plot(X, y3)

ystd = np.min(np.quantile(pred_mean, axis=1, q=0.25), axis=1)
y1 = np.min(np.quantile(pred_mean, axis=1, q=0.5), axis=1)
y3 = np.min(np.quantile(pred_mean, axis=1, q=0.75), axis=1)

axs.plot(X, ystd)
axs.plot(X, y1)
axs.plot(X, y3)

dimension = 0
q1 = np.max(np.quantile(pred_mean, axis=1, q=0.25), axis=1)
q3 = np.max(np.quantile(pred_mean, axis=1, q=0.75), axis=1)

# q1 = np.quantile(pred_mean, axis=1, q=0.25)[:, dimension]
# q3 = np.quantile(pred_mean, axis=1, q=0.75)[:, dimension]

iqr = q3 - q1
d_iqr = np.gradient(iqr)
dd_iqr = np.gradient(iqr, 2)
ddd_iqr = np.gradient(iqr, 3)

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
axs.boxplot(mu)
axs.plot(X, q1)
axs.plot(X, q3)
axs.plot(X, iqr)

iqr_ = savgol_filter(iqr, 5, 3)
d_iqr_ = np.gradient(iqr_)
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
axs.plot(X, iqr)
axs.plot(X, iqr_)
axs.plot(X, d_iqr)
axs.plot(X, d_iqr_)


#axs[1].set_xticklabels(x.tolist(), rotation='vertical')
#axs.set_title("Predictions MEAN and VAR - DIM: {}".format(dimension))

# X = [pred_var[n, :, dimension] for n in range(len(x))]
# fig = plt.figure()
# axs = plt.axes()
# axs.boxplot(X)
# axs.set_xticklabels(x, rotation='vertical')
# axs.set_title("Predictions VAR {}".format(dimension))

# X = np.abs(gt - pred_mean)
# X = [X[n, :, 0] for n in range(len(x))]
# fig = plt.figure()
# axs = plt.axes()
# axs.boxplot(X)
# axs.set_xticklabels(x, rotation='vertical')
# axs.set_title("Predictions ABS-ERROR")

plt.show()

exit()

y = [n for n in range(gt.shape[2])]
err_abs = np.abs(gt-pred_mean)
err_abs_mean = np.mean(err_abs, axis=1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes()
ax.contour(X, Y, err_abs_mean.T)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Dimensions")
# ax.set_zlabel("Error")

plt.show()

exit()











gt = data["ground_truth"]
pred_mean = data["predictions_mean"]
pred_var = data["predictions_var"]
x = data["x"].flatten()
y = [n for n in range(gt.shape[2])]
err_abs = np.abs(gt-pred_mean)
err_abs_mean = np.mean(err_abs, axis=1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, err_abs_mean.T, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Dimensions")
ax.set_zlabel("Error")

gt = data["ground_truth"]
pred_mean = data["predictions_mean"]
pred_var = data["predictions_var"]
x = data["x"].flatten()
y = [n for n in range(gt.shape[2])]
err_abs = np.abs(gt-pred_mean)
err_abs_mean = np.max(err_abs, axis=1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, err_abs_mean.T, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Dimensions")
ax.set_zlabel("Error MAX")

gt = data["ground_truth"]
pred_mean = data["predictions_mean"]
pred_var = data["predictions_var"]
x = data["x"].flatten()
y = [n for n in range(gt.shape[2])]
var_max = np.max(pred_var, axis=1)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, var_max.T, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Dimensions")
ax.set_zlabel("VAR MAX")

plt.show()

