# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import os
import sys
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

def get_model_info(model_filename):
    """Extracts the base name from a model filename."""
    return model_filename.replace("trained_model_", "").replace(".pkl", "")

def get_base_path():
    """
    Get the base path for data files. This works for both development
    and for a PyInstaller bundled application, ensuring results are saved permanently.
    """
    if getattr(sys, 'frozen', False):
        # Bundled app: models are extracted to sys._MEIPASS by PyInstaller
        return Path(sys._MEIPASS)
    else:
        # Running from source: models are under the src/ directory
        return Path(__file__).resolve().parent

def _ensure_dir(d):
    """Helper to create a directory if it doesn't exist."""
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Plotting Functions (from paper code) ---

def create_classification_plots(results_a, results_b, output_dir, label_a="Logistic Regression", label_b="Random Forest"):
    """Creates pie charts for model predictions and agreement."""
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CMSmark Classification Results', fontsize=16, fontweight='bold')
    color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'
        return my_autopct

    a_counts = results_a['Final_Prediction'].value_counts()
    axes[0].pie(a_counts.values, autopct=make_autopct(a_counts.values), startangle=90, colors=[color_map.get(c, '#BDBDBD') for c in a_counts.index], labels=a_counts.index, wedgeprops=dict(width=0.5, edgecolor='w'))
    axes[0].set_title(f'{label_a}\nCMS Distribution', fontweight='bold')

    b_counts = results_b['Final_Prediction'].value_counts()
    axes[1].pie(b_counts.values, autopct=make_autopct(b_counts.values), startangle=90, colors=[color_map.get(c, '#BDBDBD') for c in b_counts.index], labels=b_counts.index, wedgeprops=dict(width=0.5, edgecolor='w'))
    axes[1].set_title(f'{label_b}\nCMS Distribution', fontweight='bold')

    agreement = (results_a['Final_Prediction'] == results_b['Final_Prediction'])
    agreement_rate = agreement.mean()
    agreement_counts = agreement.value_counts()
    axes[2].pie([agreement_counts.get(True, 0), agreement_counts.get(False, 0)], labels=[f'Agreement ({agreement_rate:.1%})', f'Disagreement ({1-agreement_rate:.1%})'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[2].set_title('Model Agreement', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / 'classification_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Classification Results")  # ADD THIS LINE
    plt.close()

def create_confidence_boxplot(results_a, results_b, threshold, output_dir, label_a="Logistic Regression", label_b="Random Forest"):
    """Creates and saves a boxplot of prediction confidences."""
    _ensure_dir(output_dir)
    plt.figure(figsize=(8, 7))
    data_to_plot = [results_a['Max_Probability'].dropna(), results_b['Max_Probability'].dropna()]
    box = plt.boxplot(data_to_plot, labels=[label_a, label_b], patch_artist=True, medianprops=dict(color='black', linewidth=2))
    colors = ['orange', 'green']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.axhline(threshold, color='darkred', linestyle='--', linewidth=2)
    plt.text(plt.xlim()[1] * 1.01, threshold, f' Threshold ({threshold})', va='center', ha='left', color='darkred')
    plt.title('Prediction Confidence Distribution by Model', fontweight='bold', fontsize=14)
    plt.ylabel('Maximum Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = output_dir / 'confidence_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(plt.gcf(), output_path, "Confidence Boxplot")  # ADD THIS LINE
    plt.close()

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

def create_venn_diagram(results_a, results_b, output_dir, label_a="LR", label_b="RF"):
    """Create Venn diagram showing model agreement by class."""
    _ensure_dir(output_dir)
    try:
        from matplotlib_venn import venn2
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    all_classes = set(results_a['Final_Prediction'].unique()) | set(results_b['Final_Prediction'].unique())
    n_classes = len(all_classes)
    if n_classes == 0: return
    n_cols = min(3, n_classes); n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Model Agreement by CMS Class (Venn Diagrams)', fontsize=14, fontweight='bold')
    for i, cls in enumerate(sorted(all_classes)):
        ax = axes[i]
        a_samples = set(results_a[results_a['Final_Prediction'] == cls].index)
        b_samples = set(results_b[results_b['Final_Prediction'] == cls].index)
        ax.set_axis_off()
        venn = venn2([a_samples, b_samples], set_labels=(label_a, label_b), ax=ax)
        if venn.get_patch_by_id('10'): venn.get_patch_by_id('10').set_color('lightblue')
        if venn.get_patch_by_id('01'): venn.get_patch_by_id('01').set_color('lightcoral')
        if venn.get_patch_by_id('11'): venn.get_patch_by_id('11').set_color('lightgreen')
        ax.set_title(f'{cls}\n(Total: {label_a}={len(a_samples)}, {label_b}={len(b_samples)})')
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=1.5, edgecolor='black', facecolor='none'))
    for i in range(len(all_classes), len(axes)): axes[i].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / 'venn_diagram_model_agreement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Model Agreement - Venn Diagram")  # ADD THIS LINE
    plt.close()

def create_cms_heatmap(results_df, log_cpm_df, feature_names, scaler, model_name, output_dir):
    """Create heatmap showing Z-score of feature expression across CMS classes."""
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, gridspec_kw={'height_ratios': [1, 12]})
    fig.suptitle(f'CMS Classification Heatmap ({model_name}) - Top Variable Features (Z-score)', fontsize=16, fontweight='bold')
    fig.subplots_adjust(hspace=0, right=0.9)
    class_color_map = {'CMS1': '#6FCF97', 'CMS2': '#F2C185', 'CMS3': '#F28B82', 'CMS4': '#81C7E9', 'Unclassified': '#BDBDBD'}
    cms_classes = sorted([c for c in confident_samples['Final_Prediction'].unique() if c != 'Unclassified'])
    hex_colors = [class_color_map.get(cls, '#BDBDBD') for cls in sample_labels_sorted]
    sample_colors_rgb = np.array([mcolors.to_rgb(c) for c in hex_colors])
    ax1.imshow(sample_colors_rgb.reshape(1, -1, 3), aspect='auto')
    ax1.set_yticks([]); ax1.set_ylabel('CMS Class', fontweight='bold')
    legend_elements = [Line2D([0], [0], marker='s', color='w', label=cls, markerfacecolor=class_color_map.get(cls, '#BDBDBD'), markersize=10) for cls in cms_classes]
    fig.legend(handles=legend_elements, title="CMS Classes", loc='upper left', bbox_to_anchor=(0.91, 0.88), fontsize='small')
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue_white_yellow', ['#2166AC', '#FFFFFF', '#FDD835'], N=256)
    im = ax2.imshow(X_heatmap_sorted.T, cmap=custom_cmap, aspect='auto', vmin=-2, vmax=2)
    ax2.set_xlabel('Samples', fontweight='bold'); ax2.set_ylabel('Genes', fontweight='bold')
    ax2.set_yticks(range(len(feature_names_heatmap))); ax2.set_yticklabels(feature_names_heatmap, fontsize=8)
    n_samples = len(sample_names_sorted); tick_step = max(1, n_samples // 20)
    sample_tick_indices = range(0, n_samples, tick_step)
    ax2.set_xticks(sample_tick_indices); ax2.set_xticklabels([sample_names_sorted[i] for i in sample_tick_indices], rotation=45, ha='right', fontsize=8)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, extend='both')
    cbar.set_label('Z-score of Expression', rotation=270, labelpad=20, fontweight='bold')
    class_boundaries = np.where(np.diff(pd.Categorical(sample_labels_sorted).codes))[0] + 0.5
    for boundary in class_boundaries:
        ax1.axvline(boundary, color='white', linewidth=2)
        ax2.axvline(boundary, color='white', linewidth=1.5, alpha=0.9)
    output_path = output_dir / f'cms_heatmap_{model_name}.png'
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
    fig.update_layout(height=800, title_text=f'CMS Classification Heatmap ({model_name}) (Z-score)', xaxis2_title='Samples', yaxis2_title='Genes', xaxis2_showticklabels=False, legend=dict(title="CMS Classes", orientation="v", yanchor="top", y=1, xanchor="left", x=1.15))
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.write_html(output_dir / f'cms_heatmap_plotly_{model_name}.html')

def create_classification_plots_multi(results_list, labels, output_dir):
    """Creates pie charts for N model predictions + an all-model agreement chart."""
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
    """Creates a confidence boxplot for N models side by side."""
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


def create_agreement_matrix(results_list, labels, output_dir):
    """Create a pairwise agreement rate heatmap for all model combinations."""
    _ensure_dir(output_dir)
    n = len(results_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = (results_list[i]['Final_Prediction'] == results_list[j]['Final_Prediction']).mean()
    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.5)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label='Agreement Rate')
    for i in range(n):
        for j in range(n):
            txt_color = 'white' if matrix[i, j] < 0.3 or matrix[i, j] > 0.85 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1%}', ha='center', va='center',
                    fontweight='bold', fontsize=11, color=txt_color)
    ax.set_title('Pairwise Model Agreement Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    output_path = output_dir / 'model_agreement_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Model Agreement Matrix")
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

# --- Helper Functions for Inference ---

def _log_debug(results_dir, msg):
    """Append a debug message to cmsmark_debug.log in results_dir."""
    try:
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(Path(results_dir) / "cmsmark_debug.log", "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def _get_group_name(model_base):
    """Strip model type suffix to get group name."""
    for suffix in ('_lr', '_rf', '_mlp', '_xgb'):
        if model_base.endswith(suffix):
            return model_base[:-len(suffix)]
    return model_base

def _infer_models(models_dir, model_files, log_cpm_df, threshold, file_results_root):
    """Run inference for a list of model files. Returns {model_base: plotting_df}."""
    results = {}
    _log_debug(file_results_root, f"_infer_models: dir={models_dir} files={model_files}")
    for model_file in model_files:
        model_base = get_model_info(model_file)
        group_name = _get_group_name(model_base)
        try:
            model = load_pickle(models_dir / model_file)
            label_encoder = load_pickle(models_dir / f"label_encoder_{group_name}.pkl")
            feat_file = f"features_{group_name}_genes.txt" if "_pca_" in model_base else f"features_{group_name}.txt"
            feature_names = load_feature_names(models_dir / feat_file)
        except Exception as e:
            _log_debug(file_results_root, f"  SKIP load {model_base}: {type(e).__name__}: {e}")
            continue
        missing = [f for f in feature_names if f not in log_cpm_df.columns]
        if missing:
            _log_debug(file_results_root, f"  SKIP {model_base}: {len(missing)} missing features (e.g. {missing[:3]})")
            continue
        if not hasattr(model, "predict_proba"):
            _log_debug(file_results_root, f"  SKIP {model_base}: no predict_proba")
            continue
        X_input = log_cpm_df[feature_names]
        proba = model.predict_proba(X_input)
        class_names = label_encoder.classes_
        prob_df = pd.DataFrame(proba, index=X_input.index, columns=[f"{cls}_prob" for cls in class_names])
        max_proba = proba.max(axis=1)
        final_prediction = np.where(max_proba < threshold, "Unclassified", class_names[proba.argmax(axis=1)])
        model_results_df = pd.concat([pd.DataFrame({'Final_Prediction': final_prediction}, index=X_input.index), prob_df], axis=1)
        plotting_df = pd.DataFrame({'Final_Prediction': final_prediction, 'Max_Probability': max_proba}, index=X_input.index)
        group_output_dir = file_results_root / group_name
        _ensure_dir(group_output_dir)
        csv_path = group_output_dir / f"inference_results_{model_base}.csv"
        model_results_df.to_csv(csv_path)
        save_dataframe_as_html(model_results_df, csv_path, f"Inference Results - {model_base}")
        results[model_base] = plotting_df
        _log_debug(file_results_root, f"  OK {model_base}: {len(plotting_df)} samples")
        # Individual heatmaps saved directly in the group folder
        _label_map = {"_lr": "LR", "_rf": "RF", "_mlp": "MLP", "_xgb": "XGBoost"}
        model_label = next((v for k, v in _label_map.items() if model_base.endswith(k)), model_base)
        try:
            scaler = load_pickle(models_dir / f"scaler_{group_name}.pkl")
            create_cms_heatmap(plotting_df, log_cpm_df, feature_names, scaler, model_label, group_output_dir)
            create_plotly_heatmap(plotting_df, log_cpm_df, feature_names, scaler, model_label, group_output_dir)
        except FileNotFoundError:
            pass
    return results

def _create_pair_plots(group_name, df_a, df_b, models_dir, log_cpm_df, threshold, file_results_root, label_a, label_b):
    """Create comparison and visualization plots for a model pair."""
    pair_subdir = f"{label_a.replace(' ', '_')}_vs_{label_b.replace(' ', '_')}"
    group_output_dir = file_results_root / group_name / pair_subdir
    _ensure_dir(group_output_dir)
    create_classification_plots(df_a, df_b, group_output_dir, label_a, label_b)
    create_confidence_boxplot(df_a, df_b, threshold, group_output_dir, label_a, label_b)
    create_venn_diagram(df_a, df_b, group_output_dir, label_a, label_b)
    try:
        feat_file = f"features_{group_name}_genes.txt" if "_pca_" in group_name else f"features_{group_name}.txt"
        feature_names = load_feature_names(models_dir / feat_file)
        scaler = load_pickle(models_dir / f"scaler_{group_name}.pkl")
    except FileNotFoundError:
        return
    create_cms_heatmap(df_a, log_cpm_df, feature_names, scaler, label_a, group_output_dir)
    create_cms_heatmap(df_b, log_cpm_df, feature_names, scaler, label_b, group_output_dir)
    create_plotly_heatmap(df_a, log_cpm_df, feature_names, scaler, label_a, group_output_dir)
    create_plotly_heatmap(df_b, log_cpm_df, feature_names, scaler, label_b, group_output_dir)
    combined_df = pd.concat([df_a, df_b])
    combined_csv = group_output_dir / "combined_results.csv"
    combined_df.to_csv(combined_csv, index=True)
    save_dataframe_as_html(combined_df, combined_csv, "Combined Classification Results")

# --- Main Inference Function (for GUI) ---

def perform_inference(csv_path, threshold, model_group, needs_normalization):
    """Perform inference and save all results and plots."""
    base_path = get_base_path()
    input_file = Path(csv_path)
    file_results_root = input_file.parent / f"{input_file.stem}_results"
    _ensure_dir(file_results_root)

    group_to_tag = {"Excellent": "_Excellent_", "Good": "_Good_", "Excellent-Good": "_Excellent_Good_"}
    tag = group_to_tag.get(model_group)
    if not tag:
        raise ValueError(f"Unknown model group: {model_group}")

    count_df = pd.read_csv(csv_path, index_col=0)
    count_df_t = count_df.T
    total_counts = count_df_t.sum(axis=1)
    cpm_df = count_df_t.div(total_counts, axis=0) * 1_000_000
    log_cpm_df = np.log2(cpm_df + 1) if needs_normalization else cpm_df
    log_cpm_df.to_csv(file_results_root / "normalized_log_cpm.csv")
    save_dataframe_as_html(log_cpm_df, file_results_root / "normalized_log_cpm.csv", "Normalized Log CPM")

    def get_dir_files(dir_name):
        d = base_path / "models" / dir_name
        if not d.exists():
            return d, []
        files = [f for f in os.listdir(d) if f.startswith("trained_model_") and f.endswith(".pkl") and tag in f]
        if model_group == "Excellent":
            files = [f for f in files if "_Excellent_Good_" not in f]
        return d, files

    all_results = {}  # model_base -> plotting_df
    dir_map = {}      # model_base -> models_dir
    for dir_name in ("rf_lr_models", "mlp_models", "xgb_models"):
        d, files = get_dir_files(dir_name)
        _log_debug(file_results_root, f"dir={dir_name} exists={d.exists()} files={files}")
        for base, df in _infer_models(d, files, log_cpm_df, threshold, file_results_root).items():
            all_results[base] = df
            dir_map[base] = d

    if not all_results:
        raise FileNotFoundError(f"No models found for group '{model_group}'.")

    # All pairwise comparisons between every available model type
    _all_model_types = [
        ("lr",  "Logistic Regression"),
        ("rf",  "Random Forest"),
        ("mlp", "MLP"),
        ("xgb", "XGBoost"),
    ]
    for i, (type_a, label_a) in enumerate(_all_model_types):
        for type_b, label_b in _all_model_types[i + 1:]:
            groups_a = {_get_group_name(k) for k in all_results if k.endswith(f"_{type_a}")}
            groups_b = {_get_group_name(k) for k in all_results if k.endswith(f"_{type_b}")}
            for group_name in sorted(groups_a & groups_b):
                _create_pair_plots(
                    group_name,
                    all_results[f"{group_name}_{type_a}"],
                    all_results[f"{group_name}_{type_b}"],
                    dir_map[f"{group_name}_{type_a}"],
                    log_cpm_df, threshold, file_results_root, label_a, label_b
                )

    # --- All-Models Comparison (LR + RF + MLP + XGBoost together) ---
    all_model_types = [("lr", "Logistic Regression"), ("rf", "Random Forest"),
                       ("mlp", "MLP"), ("xgb", "XGBoost")]
    all_group_names = {_get_group_name(k) for k in all_results}
    for group_name in sorted(all_group_names):
        available = [(t, l) for t, l in all_model_types if f"{group_name}_{t}" in all_results]
        if len(available) < 3:
            continue  # need at least 3 models for a meaningful all-models plot
        types_avail, labels_avail = zip(*available)
        results_list = [all_results[f"{group_name}_{t}"] for t in types_avail]
        labels_list = list(labels_avail)

        all_models_dir = file_results_root / group_name / "All_Models_Comparison"
        _ensure_dir(all_models_dir)

        create_classification_plots_multi(results_list, labels_list, all_models_dir)
        create_confidence_boxplot_multi(results_list, labels_list, threshold, all_models_dir)
        create_agreement_matrix(results_list, labels_list, all_models_dir)

        # Individual heatmaps for every available model
        for t, label in available:
            model_base = f"{group_name}_{t}"
            d = dir_map[model_base]
            try:
                feat_file = (f"features_{group_name}_genes.txt" if "_pca_" in group_name
                             else f"features_{group_name}.txt")
                feature_names = load_feature_names(d / feat_file)
                scaler = load_pickle(d / f"scaler_{group_name}.pkl")
                create_cms_heatmap(all_results[model_base], log_cpm_df,
                                   feature_names, scaler, label, all_models_dir)
                create_plotly_heatmap(all_results[model_base], log_cpm_df,
                                      feature_names, scaler, label, all_models_dir)
            except FileNotFoundError:
                pass