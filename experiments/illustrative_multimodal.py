
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bezierv.classes.distfit import DistFit
import math

np.random.seed(42)

cavity_configs = [
    {'id': 1, 'mean': 9.80, 'std': 0.05, 'count': 250},
    {'id': 2, 'mean': 9.85, 'std': 0.06, 'count': 250},
    {'id': 3, 'mean': 10.10, 'std': 0.08, 'count': 250},
    {'id': 4, 'mean': 10.45, 'std': 0.05, 'count': 250}
]

all_data = []
for config in cavity_configs:
    data = np.random.normal(loc=config['mean'], scale=config['std'], size=config['count'])
    df_temp = pd.DataFrame({
        'Diameter': data,
        'Cavity': f"Cavity {config['id']}"
    })
    all_data.append(df_temp)

df = pd.concat(all_data, ignore_index=True)


plt.figure(figsize=(14, 6))

# Plot 1: Aggregate View
#plt.subplot(1, 2, 1)
#fig, ax = plt.subplots(figsize=(14, 6))

plt.hist(df['Diameter'], bins=25, color='gray', alpha=0.6, edgecolor='white', density=True)
#plt.title("Aggregate Data (All Cavities Mixed)\nAppears Multi-modal / Non-Normal")
plt.xlabel("Diameter (mm)")
plt.ylabel("Frequency")
plt.show()
#plt.savefig('cavity_all.png')

# Plot 2: Stratified View
#plt.subplot(1, 2, 2)
colors = ['blue', 'orange', 'green', 'red']
for i, config in enumerate(cavity_configs):
    cavity_data = df[df['Cavity'] == f"Cavity {config['id']}"]['Diameter']
    plt.hist(cavity_data, bins=10, alpha=0.3, color=colors[i], edgecolor='white', label=f"Cavity {config['id']}")
#plt.title("Stratified Data (Separated by Cavity)\nReveals distinct, simple distributions")
plt.xlabel("Diameter (mm)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('cavity_1_2_3_4.png')

stats = df.groupby('Cavity')['Diameter'].agg(['mean', 'std', 'count'])
aggregate_stats = pd.DataFrame({
    'Cavity': ['Aggregate'], 
    'mean': [df['Diameter'].mean()], 
    'std': [df['Diameter'].std()], 
    'count': [df['Diameter'].count()]
}).set_index('Cavity')
print("Per-Cavity Statistics:")
print(stats)
print("\nAggregate Statistics:")
print(aggregate_stats)

# Plot 3: Bézier fit to aggregate data
fitter = DistFit(df['Diameter'], n=15)
multimodal_rv, mse = fitter.fit(method="mse", algorithm="nonlinear")
print(f"Bézier fit - MSE: {mse:.6f}")

mean = multimodal_rv.get_mean()            # Compute mean
var = multimodal_rv.get_variance()  # Compute variance
#q90 = multimodal_rv.quantile(0.90)         # 90th percentile
#cdf_val =multimodal_rv.cdf_x(10)         # P(X ≤ 10)
#print(f"Bézier fit - mean: {mean:.2f}, var: {var:.2f}, 90% quantile: {q90:.2f}, CDF(10): {cdf_val:.4f}")
#print(f"Bézier fit - mean: {mean:.2f}, variance: {var:.2f}")
print(f"Bézier fit - mean: {mean:.2f}")
print(f"Bézier fit - variance: {var:.2f}")
print(f"Bézier fit - std: {math.sqrt(var):.2f}")

# Create side-by-side plots
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Create a single figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# 1. Plot the gray histogram on ax1
ax1.hist(df['Diameter'], bins=25, color='gray', alpha=0.6, edgecolor='white', density=True, label='Data Histogram')

# 2. Plot the Bézier PDF on the same graph (ax1)
# Pass show=False so it doesn't immediately display the plot
multimodal_rv.plot_pdf(ax=ax1, show=False)

# Customize primary y-axis (Histogram & PDF)
ax1.set_xlabel("Diameter (mm)")
ax1.set_ylabel("Probability Density", color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 3. Create a twin y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the Bézier CDF on the secondary axis (ax2)
multimodal_rv.plot_cdf(ax=ax2, show=False)

# Customize secondary y-axis (CDF)
ax2.set_ylabel("Cumulative Probability")
ax2.tick_params(axis='y')

# Optional: Combine legends if the internal methods generate overlapping legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend_.remove() # Remove ax2's auto-generated legend
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

#plt.title("Diameter Distribution: Histogram, Bézier PDF, and Bézier CDF")
plt.tight_layout()
plt.show()