"""Shared chart style. All chart text is English so it always renders correctly."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from . import config as C

TYPE_COLORS = {'Public': '#2a9d8f', 'Private': '#457b9d', 'Abroad': '#e9a03b', 'Unknown': '#9a9a9a'}
FIELD_COLORS = {'Medicine': '#6a4c93', 'Stomatology': '#ff7f50'}
ROUND_COLORS = dict(zip(C.EXAMS, ['#4c78a8', '#72b7b2', '#54a24b', '#e45756', '#f2a541']))
ROUND_LABEL = {'2023': '2023', '2024': '2024', '2025': '2025', '2026-R1': '2026 R1', '2026-R2': '2026 R2'}


def setup():
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'savefig.facecolor': 'white', 'axes.grid.axis': 'y',
        'legend.title_fontsize': 9, 'grid.alpha': .6})


def save(fig, name, show=True):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, f'{name}.png'), dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def pct_axis(ax, axis='y'):
    from matplotlib.ticker import PercentFormatter
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))
