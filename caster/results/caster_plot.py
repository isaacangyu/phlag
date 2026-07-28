import os
import sys
import re
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CasterPlotter:
    def __init__(self, scores_file, distribution='gaussian', data_dir='../data', topologies=None, plot_dstar=False, plot_scores=True, plot_dist=True):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir
        self.topologies = topologies
        self.plot_dstar = plot_dstar
        
        # Keep data_dir path generation fallback strictly for file tracking input workflows
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Extract metadata parameters from filename
        self.extract_filename_parameters()
        
        # Read dataset into memory from the targeted data_dir environment
        self.load_data()
        
        if self.df is not None:
            # High-contrast, vibrant, and highly distinguishable color palette for the 3 topologies
            self.topo_colors = {
                'ABBA': '#1F77B4',    # Bold Royal Blue
                'BABA': '#D62728',    # Vivid Crimson Red
                'AABB': '#2CA02C',    # Vibrant Forest Green
            }
            self.color_mapping = {}
            for col in self.df.columns:
                match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
                if match:
                    topo = match.group(1).upper()
                    self.color_mapping[col] = self.topo_colors.get(topo, '#808080')
                    # Also map cleaned topology name for melted dataframes
                    self.color_mapping[topo] = self.topo_colors.get(topo, '#808080')
                else:
                    self.color_mapping[col] = '#808080'

            # 1. Run empirical histograms with optional parametric distribution overlay
            if plot_dist:
                self.plot_caster_histograms()
                if self.plot_dstar:
                    self.plot_dstar_histogram()
            
            # 2. Scatter plot three topology scores over the genome
            if plot_scores:
                self.plot_topology_scatter()

    def extract_filename_parameters(self):
        """Extracts genomic filename and normalization token from path."""
        base_name = os.path.basename(self.scores_file)
        
        # Check if the filename ends with normalized suffix before the extension
        name_part, ext = os.path.splitext(base_name)
        if name_part.endswith('_n'):
            self.gene_name = name_part[:-2]
            self.is_normalized_file = True
        else:
            self.gene_name = name_part
            self.is_normalized_file = False

    def load_data(self):
        """Parses the tab-separated value file into a Pandas DataFrame."""
        # Attempt loading directly, or prepend data_dir prefix if the file path is isolated
        target_path = self.scores_file
        if not os.path.exists(target_path) and not os.path.isabs(target_path):
            target_path = os.path.join(self.data_dir, self.scores_file)

        try:
            self.df = pd.read_csv(target_path, sep='\t')
            print(f"Loaded {len(self.df)} windows for locus '{self.gene_name}' from: {target_path}")
            print("Detected columns:", self.df.columns.tolist())
            sns.set_theme(style="whitegrid")
        except Exception as e:
            print(f"Error reading dataset file {target_path}: {e}")
            self.df = None

    def calculate_summary_statistics(self, series):
        """Calculates summary statistics, returns and sets self.params dict for scipy.stats."""
        if self.distribution:
            dist_name = self.distribution.lower()
            if dist_name in ['gaussian', 'normal']:
                dist_name = 'norm'
            
            import scipy.stats as stats_module
            dist_class = getattr(stats_module, dist_name)
            fit_vals = dist_class.fit(series)
            
            param_names = []
            if dist_class.shapes:
                param_names.extend([s.strip() for s in dist_class.shapes.split(',')])
            param_names.extend(['loc', 'scale'])
            self.params = dict(zip(param_names, fit_vals))
        else:
            self.params = {
                'loc': series.mean(),
                'scale': series.std()
            }
        return self.params

    def plot_caster_histograms(self):
        """Generates 3 subplots (1 row x 3 columns) for ABBA, BABA, AABB, fitting Gaussian to null and alt ground truth separately."""
        norm_label = 'Normalized (Min-Max)' if self.is_normalized_file else 'Raw'

        # Determine topology columns to plot
        avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        if self.topologies is not None:
            filtered_cols = []
            for col in avg_cols:
                for t in self.topologies:
                    if t.lower() in col.lower():
                        filtered_cols.append(col)
                        break
            avg_cols = filtered_cols

        if not avg_cols:
            print("No matching topology columns found to plot.")
            return

        # Check for ground truth locus pattern in filename (e.g., ...n1n8a1n5...)
        base_name = os.path.basename(self.scores_file)
        pattern_str_match = re.search(r'((?:[an]\d+)+)', base_name)
        
        has_ground_truth = False
        y_true = np.zeros(len(self.df), dtype=int)
        
        if pattern_str_match:
            pattern_str = pattern_str_match.group(1)
            blocks = re.findall(r'([an])(\d+)', pattern_str)
            if blocks:
                has_ground_truth = True
                max_pos = self.df['pos'].max() if 'pos' in self.df.columns else 0
                step_size = self.df['pos'].diff().median() if ('pos' in self.df.columns and len(self.df) > 1) else 1000
                if np.isnan(step_size) or step_size <= 0:
                    step_size = 1000
                total_span = max_pos + step_size
                block_size_bp = total_span / len(blocks) if len(blocks) > 0 else 500000
                
                curr_pos_bp = 0
                anomaly_intervals = []
                for b_type, b_id in blocks:
                    length_bp = block_size_bp
                    if b_type == 'a':
                        anomaly_intervals.append((curr_pos_bp, curr_pos_bp + length_bp))
                    curr_pos_bp += length_bp
                    
                if 'pos' in self.df.columns:
                    positions = self.df['pos'].values
                    for idx, pos in enumerate(positions):
                        for start_bp, end_bp in anomaly_intervals:
                            if start_bp <= pos < end_bp:
                                y_true[idx] = 1
                                break

        num_plots = len(avg_cols)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
        axes = axes[0]

        import matplotlib.transforms as transforms
        from scipy.stats import norm

        for i, col in enumerate(avg_cols):
            ax = axes[i]
            match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
            topo_name = match.group(1).upper() if match else col.replace('avg*', '').replace('c*', '')

            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            vals = self.df[col].values
            xmin, xmax = vals.min(), vals.max()
            margin = (xmax - xmin) * 0.15 if xmax > xmin else 1.0
            x_grid = np.linspace(xmin - margin, xmax + margin, 200)

            if has_ground_truth:
                df_null = self.df[y_true == 0]
                df_alt = self.df[y_true == 1]

                if len(df_null) > 0:
                    sns.histplot(df_null[col], ax=ax, stat='density', element='step', kde=False, alpha=0.35, color='#2B4C7E', label='Null GT (Observed)', bins=30)
                if len(df_alt) > 0:
                    sns.histplot(df_alt[col], ax=ax, stat='density', element='step', kde=False, alpha=0.35, color='#E05638', label='Alt GT (Observed)', bins=30)

                if len(df_null) > 1:
                    mu_null, std_null = norm.fit(df_null[col])
                    pdf_null = norm.pdf(x_grid, mu_null, std_null)
                    ax.plot(x_grid, pdf_null, color='#2B4C7E', linewidth=2.2, label=f'Null Fit ($\mu={mu_null:.2f}, \sigma={std_null:.2f}$)')
                    ax.axvline(mu_null, color='#2B4C7E', linestyle='--', linewidth=1.5)
                    ax.text(mu_null, 0.90, f"$\\mu_{{null}}={mu_null:.2f}$", transform=trans, color='#2B4C7E', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

                if len(df_alt) > 1:
                    mu_alt, std_alt = norm.fit(df_alt[col])
                    pdf_alt = norm.pdf(x_grid, mu_alt, std_alt)
                    ax.plot(x_grid, pdf_alt, color='#E05638', linewidth=2.2, linestyle='--', label=f'Alt Fit ($\mu={mu_alt:.2f}, \sigma={std_alt:.2f}$)')
                    ax.axvline(mu_alt, color='#E05638', linestyle=':', linewidth=1.5)
                    ax.text(mu_alt, 0.75, f"$\\mu_{{alt}}={mu_alt:.2f}$", transform=trans, color='#E05638', fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            else:
                color = self.color_mapping.get(col, '#2B4C7E')
                sns.histplot(self.df[col], ax=ax, stat='density', element='step', kde=False, alpha=0.35, color=color, label='Observed Data', bins=30)

                mu_tot, std_tot = norm.fit(self.df[col])
                pdf_tot = norm.pdf(x_grid, mu_tot, std_tot)
                ax.plot(x_grid, pdf_tot, color=color, linewidth=2.2, label=f'Gaussian Fit ($\mu={mu_tot:.2f}, \sigma={std_tot:.2f}$)')
                ax.axvline(mu_tot, color=color, linestyle='--', linewidth=1.5)
                ax.text(mu_tot, 0.90, f"$\\mu={mu_tot:.2f}$", transform=trans, color=color, fontsize=8, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            ax.set_title(f'Topology: {topo_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{norm_label} Score')
            if i == 0:
                ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.5)

        fig.suptitle(f'Topology Distribution & Gaussian Fits: {self.gene_name}', fontsize=13, fontweight='bold')
        fig.tight_layout()

        all_avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        is_all_topologies = (len(avg_cols) == len(all_avg_cols))

        if is_all_topologies:
            suffix = "all"
        else:
            topo_names = []
            for col in avg_cols:
                match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
                if match:
                    topo_names.append(match.group(1).lower())
                else:
                    topo_names.append(col.replace('avg*', '').replace('c*', ''))
            suffix = "_".join(topo_names)

        output_dir = os.path.dirname(os.path.abspath(__file__))
        suffix_norm = "_n" if self.is_normalized_file else ""
        save_path_top = os.path.join(output_dir, f'{self.distribution}_{suffix}_{self.gene_name}{suffix_norm}.png')
        plt.savefig(save_path_top, dpi=300, bbox_inches='tight')
        print(f"Saved empirical topology distribution chart to: {save_path_top}")
        plt.close()

    def plot_dstar_histogram(self):
        """Generates empirical histogram for the D* statistic distribution."""
        dstar_cols = [c for c in self.df.columns if 'D*' in c or 'D' in c and len(c) == 1 or 'dstar' in c]
        if not dstar_cols:
            return
            
        self.dstar_col_name = dstar_cols[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=self.dstar_col_name, kde=True, color='purple', bins=50, stat='density', alpha=0.5)
        
        mean_val = self.df[self.dstar_col_name].mean()
        median_val = self.df[self.dstar_col_name].median()
        std_val = self.df[self.dstar_col_name].std()
        
        # Guidelines markers
        import matplotlib.transforms as transforms
        ax = plt.gca()
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        if not self.is_normalized_file:
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Null ILS Expectation (0.0)')
        
        plt.axvline(x=mean_val, color='darkred', linestyle='-', linewidth=2, label=None)
        plt.axvline(x=median_val, color='blue', linestyle=':', linewidth=2, label=None)
        plt.axvline(x=mean_val - std_val, color='purple', linestyle='-.', linewidth=1, label=None)
        plt.axvline(x=mean_val + std_val, color='purple', linestyle='-.', linewidth=1, label=None)

        # Inline labels for D* parameters
        plt.text(
            mean_val, 0.90, f"$\\mu_{{D^*}} = {mean_val:.4f}$", transform=trans, color='darkred',
            fontsize=8.0, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
        )
        plt.text(
            median_val, 0.83, f"$\\text{{median}}_{{D^*}} = {median_val:.4f}$", transform=trans, color='blue',
            fontsize=8.0, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
        )
        plt.text(
            mean_val + std_val, 0.76, f"$\\sigma_{{D^*}} = {std_val:.4f}$", transform=trans, color='purple',
            fontsize=7.0, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
        )
        
        plt.title(f'Genomic $D^*$ Profile: {self.gene_name}', fontsize=13)
        plt.xlabel(f'{"Normalized " if self.is_normalized_file else ""} $D^*$ Value')
        plt.ylabel('Density')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        
        # Saved explicitly to caster/results with _n suffix if normalized
        output_dir = os.path.dirname(os.path.abspath(__file__))
        suffix_norm = "_n" if self.is_normalized_file else ""
        save_path_dstar = os.path.join(output_dir, f'distributions_dstar_{self.gene_name}{suffix_norm}.png')
        plt.savefig(save_path_dstar, dpi=300)
        print(f"Saved empirical D* distribution chart to: {save_path_dstar}")
        plt.close()

    def plot_topology_scatter(self):
        """Generates a scatter plot of topology scores across genomic coordinates."""
        avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        if self.topologies is not None:
            filtered_cols = []
            for col in avg_cols:
                for t in self.topologies:
                    if t.lower() in col.lower():
                        filtered_cols.append(col)
                        break
            avg_cols = filtered_cols

        if not avg_cols:
            print("No matching topology columns found to scatter plot.")
            return

        plt.figure(figsize=(12, 6))
        
        # Rename columns to standard topology names (ABBA, BABA, AABB) for clean palette hue mapping
        rename_map = {}
        for col in avg_cols:
            match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
            if match:
                rename_map[col] = match.group(1).upper()
            else:
                rename_map[col] = col

        renamed_df = self.df.rename(columns=rename_map)
        clean_cols = [rename_map.get(c, c) for c in avg_cols]

        # Melt the dataframe for seaborn plotting
        melted_df = renamed_df.melt(id_vars=['pos'], value_vars=clean_cols, 
                                 var_name='Topology', value_name='Score')
        
        # Create scatter plot with small points and transparency
        sns.scatterplot(data=melted_df, x='pos', y='Score', hue='Topology', palette=self.topo_colors, alpha=0.6, s=12)
        
        # Draw vertical split lines and null/alt annotations if filename contains ground truth pattern
        pattern_str_match = re.search(r'((?:[an]\d+)+)', self.gene_name)
        if pattern_str_match:
            pattern_str = pattern_str_match.group(1)
            blocks = re.findall(r'([an])(\d+)', pattern_str)
            if blocks:
                block_size_bp = 500000
                curr_pos_bp = 0
                for idx, (b_type, b_id) in enumerate(blocks):
                    start_pos = curr_pos_bp
                    end_pos = curr_pos_bp + block_size_bp
                    mid_pos = (start_pos + end_pos) / 2.0
                    label_text = "null" if b_type == 'n' else "alt"
                    
                    # Draw vertical boundary line at start of block (except 0)
                    if start_pos > 0:
                        plt.axvline(x=start_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
                    
                    # Label null/alt above x-axis at bottom of plot
                    plt.text(mid_pos, 0.02, label_text, transform=plt.gca().get_xaxis_transform(),
                             ha='center', va='bottom', fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
                    
                    curr_pos_bp = end_pos
                # Vertical line at end of last block
                plt.axvline(x=curr_pos_bp, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)

        # Format names for cleaner legend and title
        plt.title(f'Genomic Topology Profile: {self.gene_name}', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('Genomic Position (pos)', fontsize=11, labelpad=8)
        plt.ylabel('Topology Score Value', fontsize=11, labelpad=8)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()
        
        # Save output to caster/results with _n suffix if normalized
        output_dir = os.path.dirname(os.path.abspath(__file__))
        suffix_norm = "_n" if self.is_normalized_file else ""
        save_path_scatter = os.path.join(output_dir, f'scores_{self.gene_name}{suffix_norm}.png')
        plt.savefig(save_path_scatter, dpi=300)
        print(f"Saved empirical topology scatter plot to: {save_path_scatter}")
        plt.show()

def read_caster_scores(path):
    pos_to_caster = {}
    with open(path, "r") as f:
        header = f.readline()
        if not header:
            return pos_to_caster
        header_parts = header.strip().split("\t") if "\t" in header else header.strip().split()
        if header_parts and header_parts[0].lower() == "file":
            pos_idx = 1
            score_indices = [2, 3, 4]
        else:
            pos_idx = 0
            score_indices = [1, 2, 3]

        for line in f:
            if not line.strip():
                continue
            values = line.strip().split("\t") if "\t" in line else line.strip().split()
            try:
                pos_key = int(values[pos_idx])
                scores = np.array([
                    float(values[score_indices[0]]), 
                    float(values[score_indices[1]]), 
                    float(values[score_indices[2]])
                ], dtype=np.float32)
                pos_to_caster[pos_key] = scores
            except (ValueError, IndexError):
                continue
    return pos_to_caster

def determine_optimal_mixtures(caster_scores_path, Y, pos_to_caster, silhouette_threshold, output_dir):
    NUM_STATES = 2
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sorted_positions = sorted(pos_to_caster.keys())
    positions_kb = np.array(sorted_positions) / 1000.0
    
    num_mixtures_matrix = np.zeros((NUM_STATES, Y.shape[-1]), dtype=int)
    
    print("\n=== K-means Clustering & Silhouette Scores ===")
    
    topology_names = ["ABBA", "BABA", "AABB"] if Y.shape[-1] == 3 else [f"Coord_{i+1}" for i in range(Y.shape[-1])]
    
    for d in range(Y.shape[-1]):
        y_d = np.array(Y[:, [d]])
        topo_name = topology_names[d]
        
        k_values = list(range(2, 11))
        scores = {}
        k_opt = 2
        max_score = -2.0
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(y_d)
            score = silhouette_score(y_d, labels)
            scores[k] = score
            if score > max_score:
                max_score = score
                k_opt = k
                
        s_2 = scores[2]
        
        # Print global table
        print(f"\nTopology / Dimension: {topo_name}")
        print(f"{'k':<5} | {'Global Silhouette Score':<25}")
        print("-" * 35)
        for k in k_values:
            marker = " *" if k == k_opt else ""
            print(f"{k:<5} | {scores[k]:<25.6f}{marker}")
        print(f"Global Optimal k* = {k_opt} (Score: {max_score:.6f})")
        print(f"Global k=2 Score = {s_2:.6f}")
        
        if s_2 > silhouette_threshold:
            print(f"s_2 ({s_2:.4f}) > threshold ({silhouette_threshold:.4f}) -> Partitioning into Null and Alternative clusters:")
            
            # Fit 2-means to partition the data
            kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
            labels_2 = kmeans_2.fit_predict(y_d)
            
            # Assign cluster labels to Null vs Alternative based on mean topology score
            mean_c0 = np.mean(y_d[labels_2 == 0])
            mean_c1 = np.mean(y_d[labels_2 == 1])
            if mean_c0 < mean_c1:
                null_lbl, alt_lbl = 0, 1
            else:
                null_lbl, alt_lbl = 1, 0
                
            y_sub_N = y_d[labels_2 == null_lbl]
            pos_sub_N = positions_kb[labels_2 == null_lbl]
            y_sub_A = y_d[labels_2 == alt_lbl]
            pos_sub_A = positions_kb[labels_2 == alt_lbl]
            
            n_N = len(y_sub_N)
            n_A = len(y_sub_A)
            
            # Save global 2-means plot (kmeans_2_{topo_name}.png)
            plt.figure(figsize=(10, 5))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(x=positions_kb, y=y_d.ravel(), hue=labels_2, palette="tab10", alpha=0.8, legend="full")
            plt.title(f"{topo_name} | Global 2-means Partitioning (0=Null, 1=Alternative)", fontsize=12, fontweight='bold')
            plt.xlabel("Position (kb)", fontsize=10)
            plt.ylabel("Normalized score", fontsize=10)
            plt.tight_layout()
            plot_path_2 = output_dir / f"kmeans_2_{topo_name}.png"
            plt.savefig(plot_path_2, dpi=150)
            plt.close()
            print(f"Saved diagnostic global partitioning plot to: {plot_path_2}")
            
            # Search for best split kn + ka = k_opt minimizing total within-cluster sum of squares (inertia)
            best_kn = 1
            best_ka = k_opt - 1
            min_inertia = float('inf')
            
            print(f"\nEvaluating partitions (kn + ka = k* = {k_opt}):")
            print(f"{'Split':<12} | {'Null Inertia':<15} | {'Alt Inertia':<15} | {'Total Inertia':<15}")
            print("-" * 65)
            
            for kn in range(1, k_opt):
                ka = k_opt - kn
                if kn > n_N or ka > n_A:
                    continue
                    
                if kn == 1:
                    w_n = float(np.sum((y_sub_N - np.mean(y_sub_N)) ** 2))
                else:
                    km_n = KMeans(n_clusters=kn, random_state=42, n_init="auto")
                    km_n.fit(y_sub_N)
                    w_n = float(km_n.inertia_)
                    
                if ka == 1:
                    w_a = float(np.sum((y_sub_A - np.mean(y_sub_A)) ** 2))
                else:
                    km_a = KMeans(n_clusters=ka, random_state=42, n_init="auto")
                    km_a.fit(y_sub_A)
                    w_a = float(km_a.inertia_)
                    
                total_w = w_n + w_a
                marker = ""
                if total_w < min_inertia:
                    min_inertia = total_w
                    best_kn = kn
                    best_ka = ka
                    marker = " *"
                    
                print(f"{kn:<2} + {ka:<2} = {k_opt:<2}  | {w_n:<15.6f} | {w_a:<15.6f} | {total_w:<15.6f}{marker}")
                
            print(f"Optimal split: Null count = {best_kn}, Alternative count = {best_ka} (Inertia: {min_inertia:.6f})")
            
            num_mixtures_matrix[0, d] = best_kn
            num_mixtures_matrix[1, d] = best_ka
            
            # Save Null sub-cluster plot if kn >= 1
            if best_kn >= 1:
                if best_kn > 1:
                    sub_kmeans = KMeans(n_clusters=best_kn, random_state=42, n_init="auto")
                    sub_labels = sub_kmeans.fit_predict(y_sub_N)
                else:
                    sub_labels = np.zeros(len(y_sub_N), dtype=int)
                
                plt.figure(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                sns.scatterplot(x=pos_sub_N, y=y_sub_N.ravel(), hue=sub_labels, palette="tab10", alpha=0.8, legend="full")
                plt.title(f"{topo_name} | Null Sub-cluster {best_kn}-means Clustering", fontsize=12, fontweight='bold')
                plt.xlabel("Position (kb)", fontsize=10)
                plt.ylabel("Normalized score", fontsize=10)
                plt.tight_layout()
                plot_path_sub_N = output_dir / f"kmeans_kstar_{topo_name}_null.png"
                plt.savefig(plot_path_sub_N, dpi=150)
                plt.close()
                print(f"Saved Null sub-cluster optimal plot to: {plot_path_sub_N}")
                
            # Save Alternative sub-cluster plot if ka >= 1
            if best_ka >= 1:
                if best_ka > 1:
                    sub_kmeans = KMeans(n_clusters=best_ka, random_state=42, n_init="auto")
                    sub_labels = sub_kmeans.fit_predict(y_sub_A)
                else:
                    sub_labels = np.zeros(len(y_sub_A), dtype=int)
                
                plt.figure(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                sns.scatterplot(x=pos_sub_A, y=y_sub_A.ravel(), hue=sub_labels, palette="tab10", alpha=0.8, legend="full")
                plt.title(f"{topo_name} | Alternative Sub-cluster {best_ka}-means Clustering", fontsize=12, fontweight='bold')
                plt.xlabel("Position (kb)", fontsize=10)
                plt.ylabel("Normalized score", fontsize=10)
                plt.tight_layout()
                plot_path_sub_A = output_dir / f"kmeans_kstar_{topo_name}_alternative.png"
                plt.savefig(plot_path_sub_A, dpi=150)
                plt.close()
                print(f"Saved Alternative sub-cluster optimal plot to: {plot_path_sub_A}")
                
            # Save combined optimal plot (kmeans_kstar_{topo_name}.png)
            plt.figure(figsize=(10, 5))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(x=positions_kb, y=y_d.ravel(), hue=labels_2, palette="tab10", alpha=0.8, legend="full")
            plt.title(f"{topo_name} | Optimal GMM Mixture Partitions (Null count={best_kn}, Alt count={best_ka})", fontsize=11, fontweight='bold')
            plt.xlabel("Position (kb)", fontsize=10)
            plt.ylabel("Normalized score", fontsize=10)
            plt.tight_layout()
            plot_path_opt = output_dir / f"kmeans_kstar_{topo_name}.png"
            plt.savefig(plot_path_opt, dpi=150)
            plt.close()
            print(f"Saved combined optimal plot to: {plot_path_opt}")
            
        else:
            m_val = max(1, int(round(k_opt / 2.0)))
            print(f"s_2 ({s_2:.4f}) <= threshold ({silhouette_threshold:.4f}) -> Use k*/2 = {m_val} mixtures for both states.")
            num_mixtures_matrix[0, d] = m_val
            num_mixtures_matrix[1, d] = m_val
            
            # Save ONLY 2-means plot, NOT kmeans_kstar plot!
            kmeans_plot = KMeans(n_clusters=2, random_state=42, n_init="auto")
            labels_plot = kmeans_plot.fit_predict(y_d)
            
            plt.figure(figsize=(10, 5))
            sns.set_theme(style="whitegrid")
            sns.scatterplot(x=positions_kb, y=y_d.ravel(), hue=labels_plot, palette="tab10", alpha=0.8, legend="full")
            plt.title(f"{topo_name} | 2-means Clustering", fontsize=12, fontweight='bold')
            plt.xlabel("Position (kb)", fontsize=10)
            plt.ylabel("Normalized score", fontsize=10)
            plt.legend(title="Cluster")
            plt.tight_layout()
            
            plot_path = output_dir / f"kmeans_2_{topo_name}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved diagnostic clustering plot to: {plot_path}")
                
    return num_mixtures_matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Empirical topology and parametric fit plotter.")
    parser.add_argument("scores_file", type=str, help="Path to CASTER scores TSV file.")
    parser.add_argument("-d", dest="distribution", type=str, default="gaussian", help="Optional parametric distribution to fit.")
    parser.add_argument("-t", dest="topologies", type=str, nargs="+", default=None, help="List of topologies to plot.")
    parser.add_argument("-s", dest="plot_dstar", action="store_true", help="Plot D* histogram.")
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=["scores", "dist"],
        default=["scores", "dist"],
        help="List of plots to generate (choices: scores, dist. Default: both)",
    )
    # GMM clustering arguments
    parser.add_argument("--silhouette-threshold", type=float, default=0.5, help="Silhouette threshold (default: 0.5).")
    parser.add_argument("-o", "--output-dir", type=str, default="test", help="Output directory for GMM plots (default: test).")
    
    args = parser.parse_args()
    
    if args.distribution == "gmm":
        pos_to_caster = read_caster_scores(args.scores_file)
        if not pos_to_caster:
            print(f"Error: could not read any valid scores from {args.scores_file}")
            sys.exit(1)
            
        sorted_positions = sorted(pos_to_caster.keys())
        raw_caster_matrix = np.stack([pos_to_caster[pos] for pos in sorted_positions], axis=0)
        
        determine_optimal_mixtures(args.scores_file, raw_caster_matrix, pos_to_caster, args.silhouette_threshold, args.output_dir)
    else:
        plot_scores = args.plot and "scores" in args.plot
        plot_dist = args.plot and "dist" in args.plot
        
        CasterPlotter(
            scores_file=args.scores_file,
            distribution=args.distribution,
            topologies=args.topologies,
            plot_dstar=args.plot_dstar,
            plot_scores=plot_scores,
            plot_dist=plot_dist,
        )
