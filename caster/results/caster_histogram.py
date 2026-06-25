import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class CasterPlotter:
    def __init__(self, scores_file, normalize='zero-one', distribution=None, data_dir='../data'):
        self.scores_file = scores_file
        self.normalize = normalize
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
            
            # 2. Conditionally overlay parametric distributions if requested
            if self.distribution:
                self.plot_distribution(dist_type=self.distribution)

    def extract_filename_parameters(self):
        """Extracts genomic filename, window size, step size, and normalization token from path."""
        base_name = os.path.basename(self.scores_file)
        
        # Regex to match: {filename}_w{windowSize}_s{stepSize}[_n].tsv
        match = re.match(r"(.+?)_w(\d+)_s(\d+)(_n)?\.(tsv|txt|csv)", base_name)
        if match:
            self.gene_name = match.group(1)
            self.window_size = match.group(2)
            self.step_size = match.group(3)
            self.is_normalized_file = bool(match.group(4))
        else:
            # Fallback values if filename doesn't match the specific C++ output structure
            self.gene_name = os.path.splitext(base_name)[0]
            self.window_size = "Unknown"
            self.step_size = "Unknown"
            self.is_normalized_file = (self.normalize == 'zero-one')

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
        """Calculates mean, median, and standard deviation for a given pandas Series."""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std()
        }

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
                stats = self.calculate_summary_statistics(self.df[col])
                color = colors[i]
                plt.axvline(x=stats['mean'], color=color, linestyle='-', linewidth=2, 
                            label=f"{col} Mean ({stats['mean']:.4f})")
                plt.axvline(x=stats['median'], color=color, linestyle=':', linewidth=2, 
                            label=f"{col} Med ({stats['median']:.4f})")
                
            plt.title(f'Topology Average Scores: {self.gene_name} (Window={self.window_size}, Step={self.step_size})', fontsize=13)
            plt.xlabel(f'{norm_label} Weight Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Saved explicitly to current directory (.)
            save_path_top = f'{self.gene_name}_w{self.window_size}_topology.png'
            plt.savefig(save_path_top, dpi=300)
            print(f"Saved empirical topology distribution chart to current directory: {save_path_top}")
            plt.show()

        # 2. Histogram for the D* Statistic Distribution
        dstar_cols = [c for c in self.df.columns if 'D*' in c or 'D' in c and len(c) == 1 or 'dstar' in c]
        if dstar_cols:
            self.dstar_col_name = dstar_cols[0]
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x=self.dstar_col_name, kde=True, color='purple', bins=50, stat='density', alpha=0.5)
            
            dstar_stats = self.calculate_summary_statistics(self.df[self.dstar_col_name])
            
            # Guidelines markers
            if not self.is_normalized_file:
                plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Null ILS Expectation (0.0)')
            plt.axvline(x=dstar_stats['mean'], color='darkred', linestyle='-', linewidth=2, 
                        label=f"D* Mean ({dstar_stats['mean']:.4f})")
            plt.axvline(x=dstar_stats['median'], color='blue', linestyle=':', linewidth=2, 
                        label=f"D* Median ({dstar_stats['median']:.4f})")
            plt.axvline(x=dstar_stats['mean'] - dstar_stats['std'], color='purple', linestyle='-.', linewidth=1, 
                        label=f"-1 Std ({dstar_stats['mean'] - dstar_stats['std']:.4f})")
            plt.axvline(x=dstar_stats['mean'] + dstar_stats['std'], color='purple', linestyle='-.', linewidth=1, 
                        label=f"+1 Std ({dstar_stats['mean'] + dstar_stats['std']:.4f})")
            
            plt.title(f'Genomic $D^*$ Profile: {self.gene_name} (Window={self.window_size}, Step={self.step_size})', fontsize=13)
            plt.xlabel(f'{"Normalized " if self.is_normalized_file else ""} $D^*$ Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Saved explicitly to current directory (.)
            save_path_dstar = f'{self.gene_name}_w{self.window_size}_dstar.png'
            plt.savefig(save_path_dstar, dpi=300)
            print(f"Saved empirical D* distribution chart to current directory: {save_path_dstar}")
            plt.show()

    def plot_distribution(self, dist_type='gaussian'):
        """Fits a parametric theoretical PDF curve overlay over the genomic D* metrics."""
        dstar_cols = [c for c in self.df.columns if 'D*' in c or 'D' in c and len(c) == 1 or 'dstar' in c]
        if not dstar_cols:
            return
            
        col_name = dstar_cols[0]
        if dist_type.lower() in ['gaussian', 'normal']:
            plt.figure(figsize=(10, 6))
            
            # Plot the raw background histogram data bars out first
            sns.histplot(data=self.df, x=col_name, stat='density', color='lightgray', bins=50, alpha=0.6, label='Observed Data')
            
            # Extract summary statistics to establish curve boundaries
            stats = self.calculate_summary_statistics(self.df[col_name])
            mu, sigma = stats['mean'], stats['std']
            
            # Create array coordinates for the continuous math function curve
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 200)
            p = norm.pdf(x, mu, sigma) 
            
            # Draw standard normal fit
            plt.plot(x, p, color='crimson', linewidth=2.5, 
                     label=f'Fitted Gaussian PDF\n($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})')
            
            plt.title(f'Gaussian Parametric Model Fit: {self.gene_name} (Window={self.window_size}, Step={self.step_size})', fontsize=13)
            plt.xlabel(f'{"Normalized " if self.is_normalized_file else ""} $D^*$ Value')
            plt.ylabel('Probability Density')
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            # Saved explicitly to current directory (.)
            save_path_fit = f'{self.gene_name}_w{self.window_size}_gaussian_fit.png'
            plt.savefig(save_path_fit, dpi=300)
            print(f"Saved continuous Gaussian distribution fit profile to current directory: {save_path_fit}")
            plt.show()
        else:
            print(f"Parametric distribution model layout type '{dist_type}' is not supported.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caster_histogram.py [path_to_caster_w{W}_s{S}.tsv] [optional_distribution: gaussian]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    dist_param = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Executing the object-oriented analysis engine pipeline
    plotter = CasterPlotter(scores_file=input_file, distribution=dist_param)