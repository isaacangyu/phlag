import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class CasterPlotter:
    def __init__(self, scores_file, distribution='gaussian', data_dir='../data', topologies=None):
        self.scores_file = scores_file
        self.distribution = distribution
        self.data_dir = data_dir
        self.topologies = topologies
        
        # Keep data_dir path generation fallback strictly for file tracking input workflows
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Extract metadata parameters from filename
        self.extract_filename_parameters()
        
        # Read dataset into memory from the targeted data_dir environment
        self.load_data()
        
        if self.df is not None:
            # 1. Run empirical histograms with optional parametric distribution overlay
            self.plot_caster_histograms()
            # self.plot_dstar_histogram()

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
        """Generates empirical histograms with central tendency vertical guidelines and optional parametric distribution overlay."""
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

        plt.figure(figsize=(12, 7))
        melted_df = self.df.melt(id_vars=['pos'], value_vars=avg_cols, 
                                 var_name='Topology', value_name='Score')
        
        # Plot empirical histogram (no KDE spline, as requested: kde=False)
        sns.histplot(data=melted_df, x='Score', hue='Topology', element='step', 
                     stat='density', common_norm=False, kde=False, alpha=0.3, bins=50)
        
        colors = sns.color_palette(n_colors=len(avg_cols))
        
        # Determine if normal distribution overlay is requested
        overlay_dist = False
        dist_class = None
        dist_name = ""
        if self.distribution:
            dist_name = self.distribution.lower()
            if dist_name in ['gaussian', 'normal']:
                dist_name = 'norm'
            try:
                import scipy.stats as stats_module
                dist_class = getattr(stats_module, dist_name)
                overlay_dist = True
            except Exception as e:
                print(f"Could not load distribution module for '{self.distribution}': {e}")

        # Fit and plot theoretical distribution and expected value / std lines
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 200)

        for i, col in enumerate(avg_cols):
            color = colors[i]
            mean_val = self.df[col].mean()
            
            if overlay_dist:
                # Fit the distribution to get parameters (e.g. loc and scale)
                self.calculate_summary_statistics(self.df[col])
                dist = dist_class(**self.params)
                p = dist.pdf(x)
                
                # Plot the theoretical PDF curve overlay
                param_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
                pdf_label = f'Fitted {col} {dist_name.capitalize()} PDF\n({param_str})'
                plt.plot(x, p, color=color, linewidth=2.5, label=pdf_label)
                
                # If normal distribution, use the fitted loc and scale for expected value and std lines
                if dist_name == 'norm':
                    fit_mean = self.params['loc']
                    fit_std = self.params['scale']
                    
                    # Expected Value (E[X] or Mean) - Plot only one line!
                    plt.axvline(x=fit_mean, color=color, linestyle='--', linewidth=2, 
                                label=f"{col} E[X] ({fit_mean:.4f})")
                    # +/- 1 Std bounds
                    plt.axvline(x=fit_mean - fit_std, color=color, linestyle=':', linewidth=1.5, 
                                label=f"{col} -1 Std ({fit_mean - fit_std:.4f})")
                    plt.axvline(x=fit_mean + fit_std, color=color, linestyle=':', linewidth=1.5, 
                                label=f"{col} +1 Std ({fit_mean + fit_std:.4f})")
                else:
                    # For other distributions, plot fitted expected value/mean if available
                    fit_mean = self.params.get('loc', mean_val)
                    plt.axvline(x=fit_mean, color=color, linestyle='--', linewidth=2, 
                                label=f"{col} Fitted E[X] ({fit_mean:.4f})")
            else:
                # No distribution overlay requested, just plot the empirical mean line
                plt.axvline(x=mean_val, color=color, linestyle='-', linewidth=2, 
                            label=f"{col} Mean ({mean_val:.4f})")
                
        title_suffix = f" & Fitted {dist_name.capitalize()} PDF" if overlay_dist else ""
        plt.title(f'Topology Average Scores{title_suffix}: {self.gene_name}', fontsize=13)
        plt.xlabel(f'{norm_label} Weight Value')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Determine filename suffix dynamically based on plotted topologies
        all_avg_cols = [c for c in self.df.columns if 'avg' in c or 'c*' in c]
        is_all_topologies = (len(avg_cols) == len(all_avg_cols))
        
        if is_all_topologies:
            suffix = "topologies"
        else:
            topo_names = []
            for col in avg_cols:
                match = re.search(r'(ABBA|BABA|AABB)', col, re.IGNORECASE)
                if match:
                    topo_names.append(match.group(1).upper())
                else:
                    topo_names.append(col.replace('avg*', '').replace('avg_', ''))
            suffix = "_".join(topo_names)

        # Saved explicitly to caster/results
        output_dir = os.path.dirname(os.path.abspath(__file__))
        save_path_top = os.path.join(output_dir, f'histogram_{suffix}_{self.gene_name}.png')
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Empirical topology and parametric fit plotter.")
    parser.add_argument("scores_file", type=str, help="Path to CASTER scores TSV file.")
    parser.add_argument("distribution", type=str, nargs="?", default="gaussian", help="Optional parametric distribution to fit (default: gaussian).")
    parser.add_argument("-t", "--topologies", type=str, nargs="+", default=None, help="List of topologies to plot (default: all).")
    
    args = parser.parse_args()
    
    # Executing the object-oriented analysis engine pipeline
    plotter = CasterPlotter(scores_file=args.scores_file, distribution=args.distribution, topologies=args.topologies)