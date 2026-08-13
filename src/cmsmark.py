# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import os
import sys
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# ---------------------------------------------------------------------------
# Frozen-app fix: ensure xgboost/VERSION exists before any pickle loads xgboost.
# The VERSION file is missing from some conda builds; xgboost/core.py reads it
# via open(os.path.dirname(__file__)/VERSION) and crashes with FileNotFoundError.
# We create it from the bundled DLL so the version matches exactly.
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    _xgb_ver_file = os.path.join(sys._MEIPASS, 'xgboost', 'VERSION')
    if not os.path.exists(_xgb_ver_file):
        import ctypes as _ctypes
        os.makedirs(os.path.dirname(_xgb_ver_file), exist_ok=True)
        _ver = '2.0.0'  # safe fallback
        try:
            _dll = os.path.join(sys._MEIPASS, 'xgboost', 'lib', 'xgboost.dll')
            _lib = _ctypes.CDLL(_dll)
            _maj, _min, _pat = _ctypes.c_int(), _ctypes.c_int(), _ctypes.c_int()
            _lib.XGBoostVersion(_ctypes.byref(_maj), _ctypes.byref(_min), _ctypes.byref(_pat))
            _ver = f'{_maj.value}.{_min.value}.{_pat.value}'
        except Exception:
            pass
        with open(_xgb_ver_file, 'w', encoding='ascii') as _f:
            _f.write(_ver)

# --- Utility Functions ---

def load_pickle(path):
    """Loads a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_names(path):
    """Loads a list of feature names from a text file."""
    with open(path) as f:
        return [line.strip() for line in f]

def get_base_path():
    """
    Base path for read-only bundled assets (the 'models' folder).

    PyInstaller's onefile mode extracts everything listed in the spec's
    `datas` (models/, icon/) to a fresh temporary directory at sys._MEIPASS
    on every launch — NOT next to the .exe. Using sys.executable's folder
    here would look for 'models' where it never gets placed.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        # If running from source, the base path is the 'src' directory.
        return Path(__file__).resolve().parent

def get_app_dir():
    """
    Base path for writable, persistent app output (the 'results' folder).

    Unlike get_base_path(), this must NOT resolve to the onefile temp
    extraction dir — sys._MEIPASS is deleted when the app closes, which
    would silently discard every run's results. This resolves to the
    folder containing the .exe itself, which persists across runs and
    travels with the app if it's copied to another computer.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent

def _ensure_dir(d):
    """Helper to create a directory if it doesn't exist."""
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Plotting Functions (from paper code) ---

def create_pca_plot(expression_df, results_df, model_name, output_dir):
    """Performs PCA and creates a scatter plot colored by prediction."""
    _ensure_dir(output_dir)
    common_samples = expression_df.index.intersection(results_df.index)
    if len(common_samples) < 2: return
    aligned_expression = expression_df.loc[common_samples]
    aligned_results = results_df.loc[common_samples]
    X_scaled = StandardScaler().fit_transform(aligned_expression)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=common_samples)
    variance_explained = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 8))
    color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}
    for cls in sorted(aligned_results['Final_Prediction'].unique()):
        idx_to_plot = aligned_results['Final_Prediction'] == cls
        plt.scatter(pc_df.loc[idx_to_plot, 'PC1'], pc_df.loc[idx_to_plot, 'PC2'], label=cls, color=color_map.get(cls, 'grey'), alpha=0.8, s=50)
    plt.title(f'Principal Component Analysis (PCA) - Predictions by {model_name}', fontweight='bold', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({variance_explained[0]:.1%})')
    plt.ylabel(f'Principal Component 2 ({variance_explained[1]:.1%})')
    plt.legend(title='Predicted CMS Class')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f'pca_plot_{model_name}.png', dpi=300)
    plt.close()

def create_cms_heatmap(results_df, log_cpm_df, feature_names, scaler, model_name, output_dir):
    """Create heatmap showing Z-score of feature expression across CMS classes.

    Figure size and font sizes are derived from the actual number of
    samples/genes being drawn (and how long the shown labels are) instead of
    a single fixed figsize, so labels stay legible and nothing gets clipped
    whether there are 10 samples or 500.
    """
    _ensure_dir(output_dir)
    from matplotlib.lines import Line2D
    confident_samples = results_df[results_df['Final_Prediction'] != 'Unclassified'].copy()
    if len(confident_samples) == 0: return
    aligned_log_cpm = log_cpm_df.loc[confident_samples.index, feature_names]
    X_confident_scaled = scaler.transform(aligned_log_cpm)
    top_n_features = min(50, len(feature_names))
    top_feature_indices = np.argsort(np.var(X_confident_scaled, axis=0))[-top_n_features:]
    X_top_features = X_confident_scaled[:, top_feature_indices]
    feature_names_heatmap = [feature_names[i] for i in top_feature_indices]
    mean_expr = np.mean(X_top_features, axis=0); std_expr = np.std(X_top_features, axis=0)
    std_expr[std_expr == 0] = 1
    X_zscore = (X_top_features - mean_expr) / std_expr
    X_zscore_clipped = np.clip(X_zscore, -2, 2)
    sample_labels = confident_samples['Final_Prediction'].values
    sort_indices = np.argsort(sample_labels)
    X_heatmap_sorted = X_zscore_clipped[sort_indices, :]
    sample_labels_sorted = sample_labels[sort_indices]
    sample_names_sorted = confident_samples.index[sort_indices]

    n_samples = len(sample_names_sorted)
    n_features = len(feature_names_heatmap)

    # Dynamic figure size: width scales with sample count (more columns to
    # render), height scales with gene count (each row needs its own
    # readable label). Clamped so very small/large inputs stay sane.
    fig_width = min(36, max(10, 6 + n_samples * 0.12))
    fig_height = min(30, max(8, 3 + n_features * 0.26))
    feature_fontsize = 8 if n_features <= 30 else (7 if n_features <= 40 else 6)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height), sharex=True,
        gridspec_kw={'height_ratios': [1, 12], 'hspace': 0.03},
        constrained_layout=True,
    )
    fig.suptitle(f'CMS Classification Heatmap ({model_name}) - Top Variable Features (Z-score)', fontsize=16, fontweight='bold')
    class_color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}
    cms_classes = sorted([c for c in confident_samples['Final_Prediction'].unique() if c != 'Unclassified'])
    hex_colors = [class_color_map.get(cls, '#BDBDBD') for cls in sample_labels_sorted]
    sample_colors_rgb = np.array([mcolors.to_rgb(c) for c in hex_colors])
    ax1.imshow(sample_colors_rgb.reshape(1, -1, 3), aspect='auto')
    ax1.set_yticks([]); ax1.set_ylabel('CMS Class', fontweight='bold')
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue_white_yellow', ['#2166AC', '#FFFFFF', '#FDD835'], N=256)
    im = ax2.imshow(X_heatmap_sorted.T, cmap=custom_cmap, aspect='auto', vmin=-2, vmax=2)
    ax2.set_xlabel('Samples', fontweight='bold'); ax2.set_ylabel('Genes', fontweight='bold')
    ax2.set_yticks(range(len(feature_names_heatmap))); ax2.set_yticklabels(feature_names_heatmap, fontsize=feature_fontsize)

    # Thin sample x-tick labels to a readable count regardless of how many
    # samples exist, shrinking the font further if the shown labels are long.
    target_labels = max(10, int(fig_width * 1.5))
    tick_step = max(1, n_samples // target_labels)
    sample_tick_indices = list(range(0, n_samples, tick_step))
    shown_labels = [str(sample_names_sorted[i]) for i in sample_tick_indices]
    max_label_len = max((len(s) for s in shown_labels), default=0)
    sample_fontsize = 8 if max_label_len <= 10 else 6
    ax2.set_xticks(sample_tick_indices)
    ax2.set_xticklabels(shown_labels, rotation=45, ha='right', fontsize=sample_fontsize)

    cbar = fig.colorbar(im, ax=ax2, shrink=0.8, extend='both')
    cbar.set_label('Z-score of Expression', rotation=270, labelpad=20, fontweight='bold')

    # Legend lives outside the axes (to the right); attaching it to ax2
    # (rather than the figure) lets constrained_layout account for it, and
    # bbox_inches='tight' at save time guarantees it's never clipped either way.
    legend_elements = [Line2D([0], [0], marker='s', color='w', label=cls, markerfacecolor=class_color_map.get(cls, '#BDBDBD'), markersize=10) for cls in cms_classes]
    ax2.legend(handles=legend_elements, title="CMS Classes", loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small', borderaxespad=0.)

    class_boundaries = np.where(np.diff(pd.Categorical(sample_labels_sorted).codes))[0] + 0.5
    for boundary in class_boundaries:
        ax1.axvline(boundary, color='white', linewidth=2)
        ax2.axvline(boundary, color='white', linewidth=1.5, alpha=0.9)
    output_path = output_dir / 'heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, f"CMS Heatmap - {model_name}")  # ADD THIS LINE
    plt.close()

def create_plotly_heatmap(results_df, log_cpm_df, feature_names, scaler, model_name, output_dir):
    """Create an interactive heatmap with Plotly and save as an HTML file."""
    _ensure_dir(output_dir)
    confident_samples = results_df[results_df['Final_Prediction'] != 'Unclassified'].copy()
    if len(confident_samples) == 0: return
    aligned_log_cpm = log_cpm_df.loc[confident_samples.index, feature_names]
    X_confident_scaled = scaler.transform(aligned_log_cpm)
    top_n_features = min(50, len(feature_names))
    top_feature_indices = np.argsort(np.var(X_confident_scaled, axis=0))[-top_n_features:]
    X_top_features = X_confident_scaled[:, top_feature_indices]
    feature_names_heatmap = [feature_names[i] for i in top_feature_indices]
    mean_expr = np.mean(X_top_features, axis=0); std_expr = np.std(X_top_features, axis=0)
    std_expr[std_expr == 0] = 1
    X_zscore = (X_top_features - mean_expr) / std_expr
    X_zscore_clipped = np.clip(X_zscore, -2, 2)
    sample_labels = confident_samples['Final_Prediction'].values
    sort_indices = np.argsort(sample_labels)
    X_heatmap_sorted = X_zscore_clipped[sort_indices, :]
    sample_labels_sorted = sample_labels[sort_indices]
    sample_names_sorted = confident_samples.index[sort_indices]
    class_color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}
    unique_classes = sorted(confident_samples['Final_Prediction'].unique())
    class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    colorscale = []
    for i, cls in enumerate(unique_classes):
        color = class_color_map.get(cls, '#BDBDBD')
        colorscale.extend([[i / len(unique_classes), color], [(i + 1) / len(unique_classes), color]])
    annotation_z = [class_to_int[cls] for cls in sample_labels_sorted]
    heatmap_trace = go.Heatmap(z=X_heatmap_sorted.T, x=sample_names_sorted, y=feature_names_heatmap, colorscale=[[0.0, '#2166AC'], [0.5, '#FFFFFF'], [1.0, '#FDD835']], zmin=-2, zmax=2, showscale=True, colorbar=dict(title="Z-score"), name="Z-score")
    annotation_trace = go.Heatmap(z=[annotation_z], x=sample_names_sorted, y=['CMS Class'], colorscale=colorscale, showscale=False, hovertext=sample_labels_sorted, hoverinfo="x+text")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.05, 0.95], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(annotation_trace, row=1, col=1)
    fig.add_trace(heatmap_trace, row=2, col=1)
    for cls in unique_classes: fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=cls, marker=dict(size=10, color=class_color_map.get(cls, '#BDBDBD'), symbol='square')))
    class_boundaries = np.where(np.diff(pd.Categorical(sample_labels_sorted).codes))[0] + 0.5
    for boundary in class_boundaries: fig.add_vline(x=boundary, line_width=2, line_color="white")
    # Height scales with the number of gene rows so labels have room; Plotly
    # itself is interactive/scrollable so this mainly avoids an overly
    # cramped default view rather than fixing a hard clipping issue.
    plot_height = max(600, 200 + 12 * len(feature_names_heatmap))
    fig.update_layout(height=plot_height, title_text=f'CMS Classification Heatmap ({model_name}) (Z-score)', xaxis2_title='Samples', yaxis2_title='Genes', xaxis2_showticklabels=False, legend=dict(title="CMS Classes", orientation="v", yanchor="top", y=1, xanchor="left", x=1.15))
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.write_html(output_dir / 'heatmap_interactive.html')

def create_agreement_matrix(model_predictions, labels, output_dir):
    """Single N-model agreement-rate heatmap for the all-models comparison
    folder: cell (i, j) is the fraction of samples where model i and model j
    produced the same Final_Prediction. This is one combined summary across
    every model, not a set of per-pair comparison files.

    Figure size and font sizes scale with the number of models and the
    length of their names so labels aren't cramped or clipped whether
    there are 2 models or several with long names (e.g. "Logistic Regression").
    """
    _ensure_dir(output_dir)
    n = len(labels)
    if n < 2:
        return
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = (model_predictions[i] == model_predictions[j]).mean()

    max_label_len = max(len(l) for l in labels)
    # Cell size grows with label length so rotated/long model names have
    # room; overall figure grows with the number of models being compared.
    cell_size = max(1.1, 0.09 * max_label_len + 0.9)
    fig_width = max(5.5, n * cell_size + 1.8)
    fig_height = max(4.5, n * cell_size + 1.2)
    label_fontsize = 10 if max_label_len <= 14 else 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fontsize)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=label_fontsize)
    cbar = fig.colorbar(im, ax=ax, label='Agreement Rate', shrink=0.85)
    cbar.ax.tick_params(labelsize=label_fontsize)
    value_fontsize = 11 if n <= 4 else 9
    for i in range(n):
        for j in range(n):
            txt_color = 'white' if matrix[i, j] < 0.3 or matrix[i, j] > 0.85 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1%}', ha='center', va='center', fontweight='bold', fontsize=value_fontsize, color=txt_color)
    ax.set_title('Model Agreement Matrix', fontweight='bold', fontsize=14, pad=14)
    output_path = output_dir / 'agreement_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Model Agreement Matrix")
    plt.close()

# --- 4-model Venn agreement diagram ---
#
# The reference implementation's create_venn_diagram() drew one venn2 panel
# per CMS class (samples that model A called this class vs. samples model B
# called this class). matplotlib_venn — the library it used — only supports
# 2- and 3-set diagrams (no venn4), and this app has no other Venn-capable
# dependency installed, so a 4th model can't be added by simply widening
# that call. Adding a new plotting dependency is out of scope for this
# change (it would also need to be added to requirements.txt and the
# PyInstaller spec's bundled packages).
#
# A true *area-proportional* 4-set Venn diagram is also not generally
# achievable — at most 3 circles can represent every possible intersection
# with proportional area, and forcing 4 circles to do so produces
# regions that misrepresent the actual overlap sizes. The standard,
# textbook-correct way to draw an exact (non-area-proportional) 4-set Venn
# diagram is 4 congruent ellipses in two mirrored, offset pairs — this
# layout is what R's VennDiagram package and the pyvenn library use, and it
# is the only configuration of 4 simple closed curves that yields all
# 2^4 - 1 = 15 non-empty intersection regions as distinct, connected areas.
# It isn't area-proportional, but it is a genuine, complete Venn diagram:
# every region is uniquely identifiable and every count is exact.
_VENN4_ELLIPSES = [
    dict(cx=0.350, cy=0.400, w=0.72, h=0.45, angle=140),
    dict(cx=0.450, cy=0.500, w=0.72, h=0.45, angle=140),
    dict(cx=0.544, cy=0.500, w=0.72, h=0.45, angle=40),
    dict(cx=0.644, cy=0.400, w=0.72, h=0.45, angle=40),
]

def _point_in_ellipse(x, y, cx, cy, w, h, angle_deg):
    """Vectorized containment test for a point (or array of points) against
    a rotated ellipse (matplotlib.patches.Ellipse convention: angle is the
    counter-clockwise rotation of the ellipse in degrees)."""
    theta = np.radians(angle_deg)
    dx = x - cx
    dy = y - cy
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    return (xr / (w / 2)) ** 2 + (yr / (h / 2)) ** 2 <= 1

def _venn4_region_centroids(resolution=300):
    """Numerically finds where each of the 15 non-empty 4-set intersection
    regions actually sits in the standard 4-ellipse layout above, by
    rasterizing the unit square and grouping points by which of the 4
    ellipses contain them. Region labels are then placed at these computed
    centroids rather than at hand-copied per-region coordinates, so a count
    can never end up rendered inside the wrong region.

    Returns {bitmask: (x, y)} for bitmask 1..15, where bit i (0-indexed)
    set means "inside ellipse i"; a bitmask is absent if that combination
    has no area in this layout.
    """
    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(xs, ys)
    membership = np.stack([
        _point_in_ellipse(xx, yy, e['cx'], e['cy'], e['w'], e['h'], e['angle'])
        for e in _VENN4_ELLIPSES
    ])
    centroids = {}
    for bitmask in range(1, 16):
        sel = np.ones_like(xx, dtype=bool)
        for i in range(4):
            in_set = bool((bitmask >> i) & 1)
            sel &= membership[i] if in_set else ~membership[i]
        if sel.any():
            centroids[bitmask] = (float(xx[sel].mean()), float(yy[sel].mean()))
    return centroids

def create_venn_agreement_diagram(results_list, labels, output_dir):
    """4-model prediction-agreement Venn diagram, one panel per CMS class —
    the direct 4-set extension of the reference implementation's per-class
    venn2 panels (see module comment above for why a hand-drawn 4-ellipse
    layout is used instead of matplotlib_venn). Each panel's regions show
    the exact count of samples that were (or were not) assigned that CMS
    class by each of the 4 models, e.g. the region where only Random
    Forest's and XGBoost's ellipses overlap holds samples both models
    called this class but Logistic Regression and MLP did not.
    """
    if len(results_list) != 4:
        return
    _ensure_dir(output_dir)
    from matplotlib.patches import Ellipse as MplEllipse, Rectangle

    all_classes = sorted(set().union(*(set(r['Final_Prediction'].unique()) for r in results_list)))
    n_classes = len(all_classes)
    if n_classes == 0:
        return
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Model Agreement by CMS Class (4-Model Venn Diagram)', fontsize=16, fontweight='bold')
    colors = ['#E07B54', '#5BA85A', '#5B8DD9', '#9B59B6']
    centroids = _venn4_region_centroids()

    for i, cls in enumerate(all_classes):
        ax = axes[i]
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        sample_sets = [set(r[r['Final_Prediction'] == cls].index) for r in results_list]

        for e, color in zip(_VENN4_ELLIPSES, colors):
            ax.add_patch(MplEllipse(xy=(e['cx'], e['cy']), width=e['w'], height=e['h'],
                                     angle=e['angle'], facecolor=color, edgecolor='black',
                                     alpha=0.35, linewidth=1.2))

        for bitmask in range(1, 16):
            if bitmask not in centroids:
                continue
            included = [sample_sets[b] for b in range(4) if (bitmask >> b) & 1]
            excluded = [sample_sets[b] for b in range(4) if not (bitmask >> b) & 1]
            region_samples = set.intersection(*included)
            for ex in excluded:
                region_samples = region_samples - ex
            cx, cy = centroids[bitmask]
            ax.text(cx, cy, str(len(region_samples)), ha='center', va='center',
                    fontsize=9, fontweight='bold')

        ax.set_title(f'{cls}\n(Total unique samples: {len(set.union(*sample_sets))})',
                     fontweight='bold', fontsize=12)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=1.5,
                                edgecolor='black', facecolor='none'))

    for j in range(len(all_classes), len(axes)):
        axes[j].set_visible(False)

    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.35, edgecolor='black') for c in colors]
    fig.legend(legend_handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / 'venn_diagram_model_agreement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Model Agreement - 4-Model Venn Diagram")
    plt.close()

def create_classification_plots_multi(results_list, labels, output_dir):
    """Pie charts of each model's CMS-class distribution plus one all-model
    agreement pie, saved as classification_results.png/html in the
    all-models comparison folder. Adapted from the reference implementation
    (cmsmark_program/src/cmsmark.py)."""
    _ensure_dir(output_dir)
    n = len(results_list)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 6))
    fig.suptitle('CMSmark Classification Results — All Models', fontsize=16, fontweight='bold')
    color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}

    def make_autopct(values):
        def my_autopct(pct):
            val = int(round(pct * sum(values) / 100.0))
            return f'{pct:.1f}%\n({val})'
        return my_autopct

    for i, (results, label) in enumerate(zip(results_list, labels)):
        counts = results['Final_Prediction'].value_counts()
        axes[i].pie(counts.values, autopct=make_autopct(counts.values), startangle=90,
                    colors=[color_map.get(c, '#BDBDBD') for c in counts.index],
                    labels=counts.index, wedgeprops=dict(width=0.5, edgecolor='w'))
        axes[i].set_title(f'{label}\nCMS Distribution', fontweight='bold')

    all_preds = pd.concat([r['Final_Prediction'].rename(l) for r, l in zip(results_list, labels)], axis=1)
    all_agree = all_preds.nunique(axis=1) == 1
    rate = all_agree.mean()
    axes[-1].pie([all_agree.sum(), (~all_agree).sum()],
                 labels=[f'All Agree ({rate:.1%})', f'Disagree ({1 - rate:.1%})'],
                 autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[-1].set_title('All-Model Agreement', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / 'classification_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Classification Results — All Models")
    plt.close()

def create_confidence_boxplot_multi(results_list, labels, threshold, output_dir):
    """Boxplot of prediction-confidence distributions for all models side by
    side, saved as confidence_boxplot.png/html in the all-models comparison
    folder. Adapted from the reference implementation
    (cmsmark_program/src/cmsmark.py)."""
    _ensure_dir(output_dir)
    palette = ['#E07B54', '#5BA85A', '#5B8DD9', '#9B59B6', '#F1C40F', '#1ABC9C']
    fig = plt.figure(figsize=(max(8, 2.5 * len(results_list)), 7))
    data_to_plot = [r['Max_Probability'].dropna() for r in results_list]
    box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                      medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(box['boxes'], palette[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.axhline(threshold, color='darkred', linestyle='--', linewidth=2)
    plt.text(plt.xlim()[1] * 1.01, threshold, f' Threshold ({threshold})',
             va='center', ha='left', color='darkred')
    plt.title('Prediction Confidence Distribution — All Models', fontweight='bold', fontsize=14)
    plt.ylabel('Maximum Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = output_dir / 'confidence_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Confidence Boxplot — All Models")
    plt.close()

def save_dataframe_as_html(df, csv_path, title="CMSMARK Results"):
    """Save a DataFrame as an HTML file with styling, preserving the index"""
    html_path = Path(csv_path).with_suffix('.html')
    
    # Convert DataFrame to HTML with index
    html_content = df.to_html(
        index=True,  # Keep the index
        classes='dataframe',
        border=0,
        escape=False
    )
    
    # Add CSS styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2e4a9e;
                margin-bottom: 20px;
            }}
            .dataframe {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            .dataframe th {{
                background-color: #2e4a9e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                border: 1px solid #1c3470;
            }}
            .dataframe td {{
                padding: 10px;
                border: 1px solid #ddd;
            }}
            .dataframe tr:hover {{
                background-color: #f0f0f0;
            }}
            .dataframe tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .info {{
                color: #666;
                font-size: 0.9em;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="info">File: {Path(csv_path).name}</div>
        {html_content}
        <div class="info">Generated by CMSMARK</div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    return html_path

def save_plot_as_html(fig, png_path, title="CMSMARK Plot"):
    """Save a matplotlib figure as an HTML file with embedded image"""
    html_path = Path(png_path).with_suffix('.html')
    
    # Convert figure to base64
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                text-align: center;
            }}
            h1 {{
                color: #2e4a9e;
                margin-bottom: 20px;
            }}
            .image-container {{
                background-color: white;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                display: inline-block;
                margin: 20px auto;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            .info {{
                color: #666;
                font-size: 0.9em;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="info">File: {Path(png_path).name}</div>
        <div class="image-container">
            <img src="data:image/png;base64,{img_base64}" alt="{title}">
        </div>
        <div class="info">Generated by CMSMARK</div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

# --- Main Inference Function (for GUI) ---

# The application only exposes and uses the "Excellent" feature group. All
# four models below were confirmed (via each Pipeline's n_features_in_ /
# feature_names_in_) to take exactly the 455 genes listed in their group's
# features_Excellent*.txt as raw input — no additional internal features are
# required beyond what's shown to the user.
FEATURE_GROUP = "Excellent"

# Maps a trained-model filename's type suffix (trained_model_<group>_<suffix>.pkl)
# to the human-readable model name used as a column header in the combined
# all-models comparison table.
_MODEL_TYPE_LABELS = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "mlp": "MLP",
    "xgb": "XGBoost",
}

def _model_label(type_suffix):
    """Human-readable model name for a trained_model_*_<suffix>.pkl suffix."""
    return _MODEL_TYPE_LABELS.get(type_suffix, type_suffix.upper())

# Where each model type's trained assets live under models/, and which
# internal group tag to use when building filenames within that directory.
# LR, RF, and MLP were all trained directly on the plain Excellent gene
# panel (group tag == FEATURE_GROUP, e.g. trained_model_Excellent_lr.pkl).
# The only XGBoost model available for this group is a PCA-reduced variant
# — its Pipeline embeds its own scaler + PCA + classifier and still takes
# the same 455 Excellent genes as raw input (verified: n_features_in_=455,
# steps=['scaler','pca','classifier']) — so it uses the "_pca_100" tagged
# files instead; there is no plain (non-PCA) XGBoost model for this group.
_MODEL_TYPE_CONFIG = {
    "lr":  {"dir": "rf_lr_models", "group_tag": FEATURE_GROUP},
    "rf":  {"dir": "rf_lr_models", "group_tag": FEATURE_GROUP},
    "mlp": {"dir": "mlp_models",   "group_tag": FEATURE_GROUP},
    "xgb": {"dir": "xgb_models",   "group_tag": f"{FEATURE_GROUP}_pca_100"},
}

def _feature_file_name(group_tag):
    """features_<group_tag>.txt, or the _genes.txt variant for PCA-tagged
    groups. A PCA-tagged model's Pipeline takes the raw gene panel as
    input and reduces it internally, so "_genes.txt" (the gene list) is
    what's needed here, not the PCA component names."""
    return f"features_{group_tag}_genes.txt" if "_pca_" in group_tag else f"features_{group_tag}.txt"

def _slugify(label):
    """Lowercase, underscore-separated folder name for a model label,
    e.g. 'Logistic Regression' -> 'logistic_regression'."""
    return label.strip().lower().replace(' ', '_').replace('-', '_')

def _load_true_classes(input_file):
    """Optional ground-truth labels for the combined table's 'True Class'
    column. The input CSV itself (genes x samples counts) has no room for
    labels, so this looks for an optional sidecar file the user may supply
    alongside their input: '<input_stem>_labels.csv', with the sample name
    as the first column and the true class as the second. Returns a Series
    indexed by sample name, or None if no such file exists — the 'True
    Class' column is only added to the combined table when this is available.
    """
    candidate = input_file.parent / f"{input_file.stem}_labels.csv"
    if not candidate.exists():
        return None
    try:
        labels_df = pd.read_csv(candidate, index_col=0)
    except Exception:
        return None
    if labels_df.shape[1] < 1:
        return None
    return labels_df.iloc[:, 0]

def build_combined_predictions_table(all_results_data, group_name, true_classes=None):
    """Build one combined table for every model in `group_name`: one row per
    sample, one column per model (plus that model's confidence), aligned by
    sample name. Replaces the old per-pair 'combined_results.csv', which
    stacked two models' rows on top of each other (each sample appearing
    once per model) instead of aligning them side by side.
    """
    prefix = f"{group_name}_"
    model_dfs = {
        model_base[len(prefix):]: df
        for model_base, df in all_results_data.items()
        if model_base.startswith(prefix)
    }
    if not model_dfs:
        return pd.DataFrame()

    # Stable, readable column order: known model types first (lr, rf, mlp,
    # xgb), then any others sorted alphabetically.
    known_order = [s for s in _MODEL_TYPE_LABELS if s in model_dfs]
    other_order = sorted(s for s in model_dfs if s not in _MODEL_TYPE_LABELS)
    type_suffixes = known_order + other_order

    all_samples = sorted(set().union(*(df.index for df in model_dfs.values())))
    combined = pd.DataFrame(index=pd.Index(all_samples, name="Sample Name"))
    if true_classes is not None:
        combined["True Class"] = true_classes.reindex(combined.index)
    for suffix in type_suffixes:
        label = _model_label(suffix)
        df = model_dfs[suffix].reindex(combined.index)
        combined[label] = df["Final_Prediction"]
        combined[f"{label} Confidence"] = df["Max_Probability"]
    return combined

def perform_inference(csv_path, threshold, needs_normalization):
    """Perform inference and save all results and plots using the Excellent
    feature group, across all four models (LR, RF, MLP, XGBoost)."""
    # --- Path Setup ---
    # Use get_base_path() ONLY for loading the bundled models
    base_path = get_base_path()
    models_root = base_path / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"Critical Error: Models directory not found at {models_root}")

    # Save results under a "results" folder inside the application's own
    # persistent directory (next to the .exe when frozen, next to src/ in
    # dev) rather than next to the input CSV. The input file can live
    # anywhere — a read-only mount, a network drive, another user's folder
    # — and this app is distributed to run on different computers, so
    # results must always land in a predictable, writable location that
    # survives after the app closes (unlike base_path when frozen, which
    # is a temp extraction dir PyInstaller deletes on exit).
    input_file = Path(csv_path)
    results_root = get_app_dir() / "results"
    file_results_root = results_root / f"{input_file.stem}_results"
    # Safely clear only this input's own results folder before regenerating,
    # so re-running the same file twice never leaves obsolete files behind
    # (e.g. a heatmap from a model that no longer produces output). This
    # never touches the input CSV, trained models, or other inputs' results.
    if file_results_root.exists():
        shutil.rmtree(file_results_root)
    _ensure_dir(file_results_root)

    count_df = pd.read_csv(csv_path, index_col=0)
    count_df_t = count_df.T
    total_counts = count_df_t.sum(axis=1)
    cpm_df = count_df_t.div(total_counts, axis=0) * 1_000_000

    # --- CONDITIONAL NORMALIZATION ---
    if needs_normalization:
        log_cpm_df = np.log2(cpm_df + 1)
    else:
        log_cpm_df = cpm_df # Use the CPM data directly if not normalizing

    log_cpm_df.to_csv(file_results_root / "normalized_log_cpm.csv")
    save_dataframe_as_html(log_cpm_df, file_results_root / "normalized_log_cpm.csv", "Normalized Log CPM")  # ADD THIS LINE

    group_output_dir = file_results_root

    all_results_data = {}   # model_base -> plotting_df (Final_Prediction, Max_Probability)
    model_dirs = {}         # model_base -> that model's own output folder
    model_viz_assets = {}   # model_base -> (feature_names, heatmap scaler)

    for type_suffix, cfg in _MODEL_TYPE_CONFIG.items():
        models_subdir = models_root / cfg["dir"]
        group_tag = cfg["group_tag"]
        model_label = _model_label(type_suffix)
        model_base = f"{FEATURE_GROUP}_{type_suffix}"
        model_file = f"trained_model_{group_tag}_{type_suffix}.pkl"
        feature_file = _feature_file_name(group_tag)

        if not models_subdir.exists():
            raise FileNotFoundError(
                f"Critical Error: Models directory not found at {models_subdir} "
                f"(required for the {model_label} model)."
            )
        try:
            model = load_pickle(models_subdir / model_file)
            label_encoder = load_pickle(models_subdir / f"label_encoder_{group_tag}.pkl")
            feature_names = load_feature_names(models_subdir / feature_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing a required asset for the {model_label} model: {e}. "
                f"Expected a trained model, label encoder, and feature list for the "
                f"'{group_tag}' group in {models_subdir}."
            ) from e

        # Guard against a feature-list/model mismatch (e.g. a stale features_*.txt)
        # before touching the input data.
        expected_n = getattr(model, "n_features_in_", None)
        if expected_n is not None and expected_n != len(feature_names):
            raise ValueError(
                f"Feature schema mismatch for the {model_label} model: the model expects "
                f"{expected_n} features but {feature_file} lists {len(feature_names)}."
            )

        # Validate the input CSV actually contains every gene the model expects,
        # in the exact set the model was trained on. The model's Pipeline was
        # fit on this feature file in this exact order, so we select columns
        # by that order (not just membership) to preserve it.
        missing_genes = [f for f in feature_names if f not in log_cpm_df.columns]
        if missing_genes:
            raise ValueError(
                f"Input CSV is missing {len(missing_genes)} of the {len(feature_names)} "
                f"'{FEATURE_GROUP}' feature genes required by the {model_label} model "
                f"(e.g. {missing_genes[:5]}). Ensure the input CSV's gene identifiers "
                f"(row index) match the feature list at {models_subdir / feature_file}."
            )
        X_input = log_cpm_df[feature_names]
        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"The {model_label} model does not support predict_proba(); "
                "it is incompatible with this inference pipeline."
            )
        proba = model.predict_proba(X_input)
        class_names = label_encoder.classes_
        prob_df = pd.DataFrame(proba, index=X_input.index, columns=[f"{cls}_prob" for cls in class_names])
        max_proba = proba.max(axis=1)
        pred_classes = class_names[proba.argmax(axis=1)]
        final_prediction = np.where(max_proba < threshold, "Unclassified", pred_classes)
        model_results_df = pd.concat([pd.DataFrame({'Final_Prediction': final_prediction}, index=X_input.index), prob_df], axis=1)
        plotting_df = pd.DataFrame({'Final_Prediction': final_prediction, 'Max_Probability': max_proba}, index=X_input.index)

        # Each model gets its own output folder (lowercase, underscore-
        # separated model name) so its inference table, heatmap, and any
        # model-specific metrics never mix with another model's files.
        model_output_dir = group_output_dir / _slugify(model_label)
        _ensure_dir(model_output_dir)
        csv_output_path = model_output_dir / "inference_table.csv"
        model_results_df.to_csv(csv_output_path)
        save_dataframe_as_html(model_results_df, csv_output_path, f"Inference Table - {model_label}")

        all_results_data[model_base] = plotting_df
        model_dirs[model_base] = model_output_dir

        # Standalone scaler for this model's heatmap z-scores (visualization
        # only — distinct from any scaling embedded in the model's own
        # Pipeline). Heatmap is skipped gracefully if this isn't available.
        try:
            heatmap_scaler = load_pickle(models_subdir / f"scaler_{group_tag}.pkl")
            model_viz_assets[model_base] = (feature_names, heatmap_scaler)
        except FileNotFoundError:
            pass

    true_classes = _load_true_classes(input_file)

    # Per-model heatmap, saved only inside that model's own folder (one
    # model at a time — not a comparison between models, so this isn't
    # pairwise output).
    for model_base, (feature_names, scaler) in model_viz_assets.items():
        type_suffix = model_base.rsplit('_', 1)[1]
        model_label = _model_label(type_suffix)
        model_df = all_results_data[model_base]
        model_output_dir = model_dirs[model_base]
        create_cms_heatmap(model_df, log_cpm_df, feature_names, scaler, model_label, model_output_dir)
        create_plotly_heatmap(model_df, log_cpm_df, feature_names, scaler, model_label, model_output_dir)

    # Genuine all-models comparison output only, in its own folder: one
    # combined table (one row per sample, one column per model) and one
    # agreement matrix across every model. No per-model heatmaps or any
    # other single-model output gets copied in here.
    all_models_dir = group_output_dir / "all_models_comparison"
    combined = build_combined_predictions_table(all_results_data, FEATURE_GROUP, true_classes)
    if not combined.empty:
        _ensure_dir(all_models_dir)
        combined_csv_path = all_models_dir / "all_model_predictions.csv"
        combined.to_csv(combined_csv_path, index=True)
        save_dataframe_as_html(combined, combined_csv_path, "All-Model Predictions Comparison")

    model_bases = list(all_results_data.keys())
    if len(model_bases) >= 2:
        agreement_labels = [_model_label(mb.rsplit('_', 1)[1]) for mb in model_bases]
        agreement_predictions = [all_results_data[mb]["Final_Prediction"] for mb in model_bases]
        agreement_results_list = [all_results_data[mb] for mb in model_bases]
        _ensure_dir(all_models_dir)
        create_agreement_matrix(agreement_predictions, agreement_labels, all_models_dir)
        create_classification_plots_multi(agreement_results_list, agreement_labels, all_models_dir)
        create_confidence_boxplot_multi(agreement_results_list, agreement_labels, threshold, all_models_dir)
        create_venn_agreement_diagram(agreement_results_list, agreement_labels, all_models_dir)

    return str(file_results_root)
