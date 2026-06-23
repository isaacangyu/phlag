import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CasterPlotter:
    def __init__(self, data_dir='../data/'):
        self.data_dir = data_dir

    def calculate_summary_statistics(self, series):
        stats = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std()
        }
        return stats

    def plot_caster_histograms(self, scores_file, normalize='zero-one', distribution=None):
        """Parses Caster window metrics, computes metrics, and plots distribution landscapes."""
        scores_path = self.data_dir + scores_file
        df = pd.read_csv(scores_path, sep='\t')

        print("Detected columns:", df.columns.tolist())
        sns.set_theme(style="whitegrid")
        
        # Determine the label normalization indicator text correctly
        norm_label = 'Normalized' if normalize == 'zero-one' else 'Raw'

        # 1. Histogram for Average Topology Scores (ABBA vs BABA vs AABB)
        avg_cols = [c for c in df.columns if 'avg' in c or 'c*' in c]
        if len(avg_cols) >= 3:
            plt.figure(figsize=(12, 7))
            
            # Melt dataframe for hue-grouped plotting compatibility
            melted_df = df.melt(id_vars=['pos'], value_vars=avg_cols, 
                                var_name='Topology', value_name='Score')
            
            # Draw primary distribution layers
            sns.histplot(data=melted_df, x='Score', hue='Topology', element='step', 
                         stat='density', common_norm=False, kde=True, alpha=0.3, bins=50)
            
            # Draw vertical markers for each specific topology category layout
            colors = sns.color_palette(n_colors=len(avg_cols))
            for i, col in enumerate(avg_cols):
                stats = self.calculate_summary_statistics(df[col])
                color = colors[i]
                
                # Plot Mean and Median markers
                plt.axvline(x=stats['mean'], color=color, linestyle='-', linewidth=2, 
                            label=f"{col} Mean ({stats['mean']:.4f})")
                plt.axvline(x=stats['median'], color=color, linestyle=':', linewidth=2, 
                            label=f"{col} Med ({stats['median']:.4f})")
                
            plt.title('Distribution of Quartet/Site Pattern Scores Across Windows', fontsize=14)
            plt.xlabel(f'{norm_label} Weight/Score Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            save_path_top = 'caster_topology_distributions.png'
            plt.savefig(save_path_top, dpi=300)
            print(f"Saved topology score distribution chart to: {save_path_top}")
            plt.show()

        # 2. Histogram for the D* Statistic Distribution
        dstar_cols = [c for c in df.columns if 'D*' in c or 'D' in c and len(c) == 1 or c == 'D*']
        if dstar_cols:
            col_name = dstar_cols[0]
            plt.figure(figsize=(10, 6))
            
            # Plot baseline background distribution
            sns.histplot(data=df, x=col_name, kde=True, color='purple', bins=50, stat='density', alpha=0.5)
            
            # Compute central summary indicators for D*
            dstar_stats = self.calculate_summary_statistics(df[col_name])
            
            # Render descriptive lines (Mean, Median, and +/- 1 Standard Deviation spans)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Null ILS Expectation (0.0)')
            plt.axvline(x=dstar_stats['mean'], color='darkred', linestyle='-', linewidth=2, 
                        label=f"D* Mean ({dstar_stats['mean']:.4f})")
            plt.axvline(x=dstar_stats['median'], color='blue', linestyle=':', linewidth=2, 
                        label=f"D* Median ({dstar_stats['median']:.4f})")
            
            # Standard Deviation reference flags
            plt.axvline(x=dstar_stats['mean'] - dstar_stats['std'], color='purple', linestyle='-.', linewidth=1, 
                        label=f"-1 Std ({dstar_stats['mean'] - dstar_stats['std']:.4f})")
            plt.axvline(x=dstar_stats['mean'] + dstar_stats['std'], color='purple', linestyle='-.', linewidth=1, 
                        label=f"+1 Std ({dstar_stats['mean'] + dstar_stats['std']:.4f})")
            
            plt.title('Genomic $D^*$ Value Distribution Over Windows', fontsize=14)
            plt.xlabel('$D^*$ Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            save_path_dstar = 'dstar_distribution.png'
            plt.savefig(save_path_dstar, dpi=300)
            print(f"Saved D* statistic distribution chart to: {save_path_dstar}")
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [path_to_tsv] [optional_distribution]")
        sys.argv = [sys.argv[0], 'slidingwindow.tsv'] # local fallback safeguard
        
    input_file = sys.argv[1]
    dist_param = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize implementation engine class
    plotter = CasterPlotter()
    plotter.plot_caster_histograms(scores_file=input_file, distribution=dist_param)