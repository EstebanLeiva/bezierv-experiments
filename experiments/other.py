import numpy as np
from bezierv.classes.distfit import DistFit

# Synthetic bounded data
rng = np.random.default_rng(42)
data = rng.beta(2, 5, 1000)  # replace with your data

fitter = DistFit(data, n=4)            # n = degree (n control segments, n+1 control points)
bz, mse = fitter.fit(method="nonlinear")  # or: 'nonlinear', 'projsubgrad', 'neldermead'
print("MSE:", mse)

samples = bz.random(10_000, rng=42)     # draw samples via inverse CDF
q90 = bz.quantile(0.90)                 # 90% quantile
print("90% quantile:", q90)

bz.plot_cdf(data)