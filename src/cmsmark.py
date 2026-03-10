# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

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
        # If the app is running as a bundled executable, the base path is
        # the directory containing the .exe file.
        return Path(sys.executable).parent
    else:
        # If running from source, the base path is the 'src' directory.
        return Path(__file__).resolve().parent

def _ensure_dir(d):
    """Helper to create a directory if it doesn't exist."""
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Plotting Functions (from paper code) ---

def create_classification_plots(lr_results, rf_results, output_dir):
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

    lr_counts = lr_results['Final_Prediction'].value_counts()
    axes[0].pie(lr_counts.values, autopct=make_autopct(lr_counts.values), startangle=90, colors=[color_map.get(c, '#BDBDBD') for c in lr_counts.index], labels=lr_counts.index, wedgeprops=dict(width=0.5, edgecolor='w'))
    axes[0].set_title('Logistic Regression\nCMS Distribution', fontweight='bold')

    rf_counts = rf_results['Final_Prediction'].value_counts()
    axes[1].pie(rf_counts.values, autopct=make_autopct(rf_counts.values), startangle=90, colors=[color_map.get(c, '#BDBDBD') for c in rf_counts.index], labels=rf_counts.index, wedgeprops=dict(width=0.5, edgecolor='w'))
    axes[1].set_title('Random Forest\nCMS Distribution', fontweight='bold')

    agreement = (lr_results['Final_Prediction'] == rf_results['Final_Prediction'])
    agreement_rate = agreement.mean()
    agreement_counts = agreement.value_counts()
    axes[2].pie([agreement_counts.get(True, 0), agreement_counts.get(False, 0)], labels=[f'Agreement ({agreement_rate:.1%})', f'Disagreement ({1-agreement_rate:.1%})'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[2].set_title('Model Agreement', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / 'classification_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    save_plot_as_html(fig, output_path, "Classification Results")  # ADD THIS LINE
    plt.close()

def create_confidence_boxplot(lr_results, rf_results, threshold, output_dir):
    """Creates and saves a boxplot of prediction confidences."""
    _ensure_dir(output_dir)
    plt.figure(figsize=(8, 7))
    data_to_plot = [lr_results['Max_Probability'].dropna(), rf_results['Max_Probability'].dropna()]
    box = plt.boxplot(data_to_plot, labels=['Logistic Regression', 'Random Forest'], patch_artist=True, medianprops=dict(color='black', linewidth=2))
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

def create_venn_diagram(lr_results, rf_results, output_dir):
    """Create Venn diagram showing model agreement by class."""
    _ensure_dir(output_dir)
    try:
        from matplotlib_venn import venn2
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    all_classes = set(lr_results['Final_Prediction'].unique()) | set(rf_results['Final_Prediction'].unique())
    n_classes = len(all_classes)
    if n_classes == 0: return
    n_cols = min(3, n_classes); n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Model Agreement by CMS Class (Venn Diagrams)', fontsize=14, fontweight='bold')
    for i, cls in enumerate(sorted(all_classes)):
        ax = axes[i]
        lr_samples = set(lr_results[lr_results['Final_Prediction'] == cls].index)
        rf_samples = set(rf_results[rf_results['Final_Prediction'] == cls].index)
        ax.set_axis_off()
        venn = venn2([lr_samples, rf_samples], set_labels=('LR', 'RF'), ax=ax)
        if venn.get_patch_by_id('10'): venn.get_patch_by_id('10').set_color('lightblue')
        if venn.get_patch_by_id('01'): venn.get_patch_by_id('01').set_color('lightcoral')
        if venn.get_patch_by_id('11'): venn.get_patch_by_id('11').set_color('lightgreen')
        ax.set_title(f'{cls}\n(Total: LR={len(lr_samples)}, RF={len(rf_samples)})')
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

def perform_inference(csv_path, threshold, model_group, needs_normalization): # Add new argument
    """Perform inference and save all results and plots."""
    # --- Path Setup ---
    # Use get_base_path() ONLY for loading the bundled models
    base_path = get_base_path()
    models_subdir = base_path / "models" / "rf_lr_models"
    if not models_subdir.exists():
        raise FileNotFoundError(f"Critical Error: Models directory not found at {models_subdir}")

    # *** NEW: Save results in a folder next to the input CSV file ***
    input_file = Path(csv_path)
    file_results_root = input_file.parent / f"{input_file.stem}_results"
    _ensure_dir(file_results_root)

    # --- Model Selection Logic ---
    all_model_files = [f for f in os.listdir(models_subdir) if f.startswith("trained_model_") and f.endswith(".pkl")]
    group_to_tag = {"Excellent": "_Excellent_", "Good": "_Good_", "Excellent-Good": "_Excellent_Good_"}
    tag = group_to_tag.get(model_group)
    if not tag: raise ValueError(f"Unknown model group: {model_group}")
    model_files = [f for f in all_model_files if tag in f]
    if model_group == "Excellent": model_files = [f for f in model_files if "_Excellent_Good_" not in f]
    if not model_files: raise FileNotFoundError(f"No models found for group '{model_group}'.")

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

    all_results_data = {}
    for model_file in model_files:
        model_base = get_model_info(model_file)
        group_name = model_base.replace('_lr','').replace('_rf','')
        try:
            model = load_pickle(models_subdir / model_file)
            label_encoder = load_pickle(models_subdir / f"label_encoder_{group_name}.pkl")
            if "_pca_" in model_base:
                feature_names = load_feature_names(models_subdir / f"features_{group_name}_genes.txt")
            else:
                feature_names = load_feature_names(models_subdir / f"features_{group_name}.txt")
        except FileNotFoundError:
            continue
        missing_genes = [f for f in feature_names if f not in log_cpm_df.columns]
        if missing_genes: continue
        X_input = log_cpm_df[feature_names]
        if not hasattr(model, "predict_proba"): continue
        proba = model.predict_proba(X_input)
        class_names = label_encoder.classes_
        prob_df = pd.DataFrame(proba, index=X_input.index, columns=[f"{cls}_prob" for cls in class_names])
        max_proba = proba.max(axis=1)
        pred_classes = class_names[proba.argmax(axis=1)]
        final_prediction = np.where(max_proba < threshold, "Unclassified", pred_classes)
        model_results_df = pd.concat([pd.DataFrame({'Final_Prediction': final_prediction}, index=X_input.index), prob_df], axis=1)
        plotting_df = pd.DataFrame({'Final_Prediction': final_prediction, 'Max_Probability': max_proba}, index=X_input.index)
        group_output_dir = file_results_root / group_name
        _ensure_dir(group_output_dir)
        csv_output_path = group_output_dir / f"inference_results_{model_base}.csv"
        model_results_df.to_csv(csv_output_path)
        save_dataframe_as_html(model_results_df, csv_output_path, f"Inference Results - {model_base}")  # ADD THIS LINE

        all_results_data[model_base] = plotting_df

    group_names = sorted(list(set(k.replace('_lr','').replace('_rf','') for k in all_results_data.keys())))
    for group_name in group_names:
        group_output_dir = file_results_root / group_name
        lr_df = all_results_data.get(f"{group_name}_lr")
        rf_df = all_results_data.get(f"{group_name}_rf")
        if lr_df is None or rf_df is None: continue
        create_classification_plots(lr_df, rf_df, group_output_dir)
        create_confidence_boxplot(lr_df, rf_df, threshold, group_output_dir)
        create_venn_diagram(lr_df, rf_df, group_output_dir)
        try:
            if "_pca_" in group_name:
                feature_names = load_feature_names(models_subdir / f"features_{group_name}_genes.txt")
            else:
                feature_names = load_feature_names(models_subdir / f"features_{group_name}.txt")
            scaler = load_pickle(models_subdir / f"scaler_{group_name}.pkl")
        except FileNotFoundError:
            continue
        create_cms_heatmap(lr_df, log_cpm_df, feature_names, scaler, "LR", group_output_dir)
        create_cms_heatmap(rf_df, log_cpm_df, feature_names, scaler, "RF", group_output_dir)
        create_plotly_heatmap(lr_df, log_cpm_df, feature_names, scaler, "LR", group_output_dir)
        create_plotly_heatmap(rf_df, log_cpm_df, feature_names, scaler, "RF", group_output_dir)
        results_df = pd.concat([lr_df, rf_df], axis=0)
        combined_csv_path = group_output_dir / "combined_results.csv"
        results_df.to_csv(combined_csv_path, index=True)  # Make sure index=True
        save_dataframe_as_html(results_df, combined_csv_path, "Combined Classification Results")  # This line already exists but verify it's there
