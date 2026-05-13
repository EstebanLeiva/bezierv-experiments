import math
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bezierv.classes.distfit import DistFit

# ── Publication-quality style (Basem Rajjoub / Okabe–Ito) ────────────────────
mpl.rcParams.update({
    'font.family'         : 'serif',
    'font.size'           : 10,
    'axes.labelsize'      : 10,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 9,
    'axes.prop_cycle'     : mpl.cycler('color', [
        '#0072B2', '#D55E00', '#009E73',
        '#E69F00', '#CC79A7', '#56B4E9',
    ]),
    'lines.linewidth'     : 1.5,
    'axes.linewidth'      : 0.8,
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'xtick.major.size'    : 4,
    'ytick.major.size'    : 4,
    'xtick.minor.size'    : 2,
    'ytick.minor.size'    : 2,
    'xtick.major.width'   : 0.8,
    'ytick.major.width'   : 0.8,
    'xtick.minor.width'   : 0.6,
    'ytick.minor.width'   : 0.6,
    'lines.markersize'    : 4,
    'errorbar.capsize'    : 3,
    'axes.xmargin'        : 0.02,
    'axes.ymargin'        : 0.02,
    'legend.frameon'      : False,
    'savefig.bbox'        : 'tight',
    'savefig.dpi'         : 300,
    **(
        {'text.usetex'        : True,
         'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
         'pgf.texsystem'      : 'pdflatex',
         'pgf.rcfonts'        : False}
        if shutil.which('latex') else
        {'text.usetex'        : False,
         'mathtext.fontset'   : 'cm'}
    ),
})

mm = 1 / 25.4

# ── Data generation ───────────────────────────────────────────────────────────
np.random.seed(42)

cavity_configs = [
    {'id': 1, 'mean': 9.80,  'std': 0.05, 'count': 250},
    {'id': 2, 'mean': 9.85,  'std': 0.06, 'count': 250},
    {'id': 3, 'mean': 10.10, 'std': 0.08, 'count': 250},
    {'id': 4, 'mean': 10.45, 'std': 0.05, 'count': 250},
]

all_data = []
for config in cavity_configs:
    data = np.random.normal(loc=config['mean'], scale=config['std'], size=config['count'])
    all_data.append(pd.DataFrame({'Diameter': data, 'Cavity': f"Cavity {config['id']}"}))

df = pd.concat(all_data, ignore_index=True)

# ── Plot 1: Aggregate histogram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(130 * mm, 90 * mm), layout='constrained')
ax.hist(df['Diameter'], bins=25, color='#BBBBBB', edgecolor='white', density=True)
ax.set_xlabel(r'Diameter (mm)')
ax.set_ylabel(r'Probability density')
ax.spines[['top', 'right']].set_visible(False)
plt.show()

# ── Plot 2: Stratified histogram ──────────────────────────────────────────────
okabe_ito = ['#0072B2', '#D55E00', '#009E73', '#E69F00']

fig, ax = plt.subplots(figsize=(130 * mm, 90 * mm), layout='constrained')
for i, config in enumerate(cavity_configs):
    cavity_data = df[df['Cavity'] == f"Cavity {config['id']}"]['Diameter']
    ax.hist(cavity_data, bins=10, alpha=0.5, color=okabe_ito[i],
            edgecolor='white', label=f"Cavity {config['id']}")
ax.set_xlabel(r'Diameter (mm)')
ax.set_ylabel(r'Count')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()

# ── Statistics ────────────────────────────────────────────────────────────────
stats = df.groupby('Cavity')['Diameter'].agg(['mean', 'std', 'count'])
aggregate_stats = pd.DataFrame({
    'Cavity': ['Aggregate'],
    'mean'  : [df['Diameter'].mean()],
    'std'   : [df['Diameter'].std()],
    'count' : [df['Diameter'].count()],
}).set_index('Cavity')

print("Per-Cavity Statistics:")
print(stats)
print("\nAggregate Statistics:")
print(aggregate_stats)

# ── Bézier fit ────────────────────────────────────────────────────────────────
fitter = DistFit(df['Diameter'], n=15)
multimodal_rv, mse = fitter.fit(method='mse', algorithm='nonlinear')
print(f"\nBézier fit — MSE: {mse:.6f}")

mean = multimodal_rv.get_mean()
var  = multimodal_rv.get_variance()
print(f"Bézier fit — mean: {mean:.4f}")
print(f"Bézier fit — std:  {math.sqrt(var):.4f}")

# ── Plot 3: Histogram + Bézier PDF + Bézier CDF ───────────────────────────────
x_vals  = np.linspace(df['Diameter'].min(), df['Diameter'].max(), 400)
pdf_vals = np.array([multimodal_rv.pdf_x(x) for x in x_vals])
cdf_vals = np.array([multimodal_rv.cdf_x(x) for x in x_vals])

C_HIST = '#BBBBBB'   # neutral gray
C_PDF  = '#0072B2'   # Okabe–Ito blue
C_CDF  = '#D55E00'   # Okabe–Ito vermillion

fig, ax1 = plt.subplots(figsize=(140 * mm, 95 * mm), layout='constrained')

# Histogram (density)
ax1.hist(df['Diameter'], bins=25, color=C_HIST, edgecolor='white',
         density=True, alpha=0.85, label='Data', zorder=1)

# Bézier PDF
ax1.plot(x_vals, pdf_vals, color=C_PDF, label=r'B\'ezier PDF', zorder=2)

ax1.set_xlabel(r'Value')
ax1.set_ylabel(r'Probability density')
ax1.spines[['top']].set_visible(False)

# Bézier CDF on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(x_vals, cdf_vals, color=C_CDF, linestyle='--',
         label=r'B\'ezier CDF', zorder=2)
ax2.set_ylabel(r'Cumulative probability')
ax2.set_ylim(-0.02, 1.08)
ax2.spines[['top']].set_visible(False)

# Unified legend placed below the figure
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='outside lower center', ncol=3)

plt.show()
fig.savefig('multimodal.pgf', bbox_inches=None)