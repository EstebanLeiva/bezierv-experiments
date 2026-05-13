"""Publication-quality matplotlib rcParams used across all paper figures."""
import shutil
from pathlib import Path

import matplotlib as mpl

mm = 1 / 25.4


def save(fig, path, *, dpi: int = 200) -> Path:
    """Save *fig* as the given .pgf and an accompanying .png preview.

    The .pgf is what gets embedded in the paper; the .png is for quickly
    eyeballing the figure without compiling LaTeX. Returns the .pgf path.
    """
    path = Path(path)
    fig.savefig(path, bbox_inches=None)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    return path


def apply():
    """Apply the paper's rcParams to the global matplotlib state."""
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
