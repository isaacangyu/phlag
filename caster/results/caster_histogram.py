import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class CasterPlotter:
    def __init__(self, scores_file, distribution='gaussian', data_dir='../data'):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir
        
        # Keep data_dir path generation fallback strictly for file tracking input workflows
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Extract metadata parameters from filename
        self.extract_filename_parameters()
        
        # Read dataset into memory from the targeted data_dir environment
        self.load_data()
        
        if self.df is not None:
            # 1. Run empirical histograms with vertical statistic indicators
            self.plot_caster_histograms()
            # self.plot_dstar_histogram()
            
            # 2. Conditionally overlay parametric distributions if requested
            if self.distribution:
                self.plot_distribution(target='topology')

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
        """Generates empirical histograms with central tendency vertical guidelines."""
        norm_label = 'Normalized (Min-Max)' if self.is_normalized_file else 'Raw'

        # 1. Histogram for Average Topology Scores (ABBA vs BABA vs AABB)
        avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        if len(avg_cols) >= 3:
            plt.figure(figsize=(12, 7))
            melted_df = self.df.melt(id_vars=['pos'], value_vars=avg_cols, 
                                     var_name='Topology', value_name='Score')
            
            sns.histplot(data=melted_df, x='Score', hue='Topology', element='step', 
                         stat='density', common_norm=False, kde=True, alpha=0.3, bins=50)
            
            colors = sns.color_palette(n_colors=len(avg_cols))
            for i, col in enumerate(avg_cols):
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                color = colors[i]
                plt.axvline(x=mean_val, color=color, linestyle='-', linewidth=2, 
                            label=f"{col} Mean ({mean_val:.4f})")
                plt.axvline(x=median_val, color=color, linestyle=':', linewidth=2, 
                            label=f"{col} Med ({median_val:.4f})")
                
            plt.title(f'Topology Average Scores: {self.gene_name}', fontsize=13)
            plt.xlabel(f'{norm_label} Weight Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Saved explicitly to caster/results
            output_dir = os.path.dirname(os.path.abspath(__file__))
            save_path_top = os.path.join(output_dir, f'histogram_topology_{self.gene_name}.png')
            plt.savefig(save_path_top, dpi=300)
            print(f"Saved empirical topology distribution chart to: {save_path_top}")
            plt.show()

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
        if not self.is_normalized_file:
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Null ILS Expectation (0.0)')
        plt.axvline(x=mean_val, color='darkred', linestyle='-', linewidth=2, 
                    label=f"D* Mean ({mean_val:.4f})")
        plt.axvline(x=median_val, color='blue', linestyle=':', linewidth=2, 
                    label=f"D* Median ({median_val:.4f})")
        plt.axvline(x=mean_val - std_val, color='purple', linestyle='-.', linewidth=1, 
                    label=f"-1 Std ({mean_val - std_val:.4f})")
        plt.axvline(x=mean_val + std_val, color='purple', linestyle='-.', linewidth=1, 
                    label=f"+1 Std ({mean_val + std_val:.4f})")
        
        plt.title(f'Genomic $D^*$ Profile: {self.gene_name}', fontsize=13)
        plt.xlabel(f'{"Normalized " if self.is_normalized_file else ""} $D^*$ Value')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Saved explicitly to caster/results
        output_dir = os.path.dirname(os.path.abspath(__file__))
        save_path_dstar = os.path.join(output_dir, f'histogram_dstar_{self.gene_name}.png')
        plt.savefig(save_path_dstar, dpi=300)
        print(f"Saved empirical D* distribution chart to: {save_path_dstar}")
        plt.show()

    def plot_distribution(self, target='topology'):
        """Fits a parametric theoretical PDF curve overlay over either genomic D* metrics or average topology scores."""
        if target == 'dstar':
            cols_to_plot = [c for c in self.df.columns if 'D*' in c or 'D' in c and len(c) == 1 or 'dstar' in c]
        elif target == 'topology':
            cols_to_plot = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        else:
            print(f"Unknown target: {target}. Must be 'dstar' or 'topology'.")
            return
            
        if not cols_to_plot:
            return
            
        if self.distribution:
            plt.figure(figsize=(12, 7) if target == 'topology' else (10, 6))
            
            # Create the distribution from scipy.stats with the params dict
            dist_name = self.distribution.lower()
            if dist_name in ['gaussian', 'normal']:
                dist_name = 'norm'
                
            import scipy.stats as stats_module
            dist_class = getattr(stats_module, dist_name)
            
            # Choose colors
            colors = sns.color_palette(n_colors=len(cols_to_plot))
            
            # Plot the background histogram(s)
            if target == 'topology':
                for i, col in enumerate(cols_to_plot):
                    sns.histplot(data=self.df, x=col, stat='density', color=colors[i], bins=50, alpha=0.15, label=f'Observed {col}')
            else:
                col_name = cols_to_plot[0]
                sns.histplot(data=self.df, x=col_name, stat='density', color='lightgray', bins=50, alpha=0.6, label='Observed Data')
            
            # Create array coordinates for the continuous math function curve
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 200)
            
            # Fit and plot PDF overlay for each target column
            for i, col in enumerate(cols_to_plot):
                self.calculate_summary_statistics(self.df[col])
                dist = dist_class(**self.params)
                p = dist.pdf(x)
                
                param_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
                color = colors[i] if target == 'topology' else 'crimson'
                label = f'Fitted {col} {dist_name.capitalize()} PDF\n({param_str})' if target == 'topology' else f'Fitted {dist_name.capitalize()} PDF\n({param_str})'
                
                plt.plot(x, p, color=color, linewidth=2.5, label=label)
            
            title_suffix = "Topology Scores" if target == 'topology' else "Genomic $D^*$ Value"
            plt.title(f'{dist_name.capitalize()} Parametric Model Fit: {self.gene_name} ({title_suffix})', fontsize=13)
            plt.xlabel(f'{"Normalized " if self.is_normalized_file else ""} Weight Value' if target == 'topology' else f'{"Normalized " if self.is_normalized_file else ""} $D^*$ Value')
            plt.ylabel('Probability Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') if target == 'topology' else plt.legend(loc='upper left')
            plt.tight_layout()
            
            # Saved explicitly to caster/results
            output_dir = os.path.dirname(os.path.abspath(__file__))
            save_path_fit = os.path.join(output_dir, f'distribution_{target}_{self.gene_name}_{dist_name}_fit.png')
            plt.savefig(save_path_fit, dpi=300)
            print(f"Saved continuous {dist_name.capitalize()} distribution fit profile to: {save_path_fit}")
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caster_histogram.py [path_to_caster_w{W}_s{S}.tsv] [optional_distribution: gaussian]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    dist_param = sys.argv[2] if len(sys.argv) > 2 else 'gaussian'
    
    # Executing the object-oriented analysis engine pipeline
    plotter = CasterPlotter(scores_file=input_file, distribution=dist_param)