import sys
import pathlib
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from phlag.main import Phlag, parse_arguments

class PhlagPlotter:
    """
    Object-oriented plotter for visualizing the probability distributions
    of standard multi-species coalescent (MSC) background (State 0, Null) vs.
    alternative/anomalous (State 1, Alternative) states before and after EM fitting.
    """
    def __init__(self, phlag, initial_means, initial_covariances):
        self.phlag = phlag
        self.initial_means = initial_means
        self.initial_covariances = initial_covariances
        
        # Ingest and configure metadata parameters
        self.extract_metadata()
        
        # Generate the visual distribution charts
        self.plot_distributions()

    def extract_metadata(self):
        """Extracts genomic filename, dimension, and styles configuration."""
        self.input_path = pathlib.Path(self.phlag.args.caster_scores)
        self.emission_dim = self.phlag.Y.shape[-1]
        
        # Map coordinates to topology names if we have the standard 3 topologies
        if not self.phlag.ilr_transform and self.emission_dim == 3:
            self.topology_names = ["ABBA", "BABA", "AABB"]
        else:
            self.topology_names = [
                f"ILR Coord {i+1}" if self.phlag.ilr_transform else f"Coord {i+1}" 
                for i in range(self.emission_dim)
            ]

        # Define premium color palette and labels
        self.colors = {
            0: {
                'line': '#2B4C7E',    # Deep Steel Blue
                'fill': '#2B4C7E',
                'label': 'State 0 (Null)'
            },
            1: {
                'line': '#E05A47',    # Warm Coral
                'fill': '#E05A47',
                'label': 'State 1 (Alternative)'
            }
        }

    def plot_distributions(self):
        """Prepares the subplot layout and runs plotting for Before and After EM states."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
            'axes.edgecolor': '#cccccc',
            'grid.color': '#f0f0f0'
        })
        
        # Create a 2x3 panel layout (or 2x1 if single dimension)
        fig, axes = plt.subplots(2, self.emission_dim, figsize=(5 * self.emission_dim, 9.0), sharey=False)
        
        if self.emission_dim == 1:
            axes = np.array([[axes[0]], [axes[1]]])
            
        # Determine consistent coordinate limits across Before/After plots
        ranges = {}
        for d in range(self.emission_dim):
            ymin, ymax = float(np.min(self.phlag.Y[:, d])), float(np.max(self.phlag.Y[:, d]))
            ypad = (ymax - ymin) * 0.20 or 0.1
            ranges[d] = np.linspace(ymin - ypad, ymax + ypad, 300)

        # Plot Row 0: Before EM (Initial theoretical setup)
        self._plot_row(axes[0], self.initial_means, self.initial_covariances, ranges, title_prefix="Before EM", plot_empirical=False)
        
        # Plot Row 1: After EM (Fitted theoretical setup and assigned empirical data)
        final_means = np.array(self.phlag.params.emissions.means)
        final_covariances = np.array(self.phlag.params.emissions.covariances)
        self._plot_row(axes[1], final_means, final_covariances, ranges, title_prefix="After EM", plot_empirical=True)
        
        plt.tight_layout()
        self.save_plot()

    def _plot_row(self, row_axes, means, covariances, ranges, title_prefix, plot_empirical=False):
        """Plots a single row (either Before EM or After EM) across all topologies."""
        if plot_empirical:
            most_likely_states = self.phlag.hmm.most_likely_states(self.phlag.params, self.phlag.Y)
            
        for d in range(self.emission_dim):
            ax = row_axes[d]
            x_vals = ranges[d]
            
            import matplotlib.transforms as transforms
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            
            # 1. Plot empirical step-histograms for Assigned State Data points
            if plot_empirical:
                y_state0 = np.array(self.phlag.Y[most_likely_states == 0, d])
                y_state1 = np.array(self.phlag.Y[most_likely_states == 1, d])
                
                if len(y_state0) > 0:
                    sns.histplot(
                        y_state0, ax=ax, color=self.colors[0]['fill'], 
                        stat="density", kde=False, alpha=0.12, 
                        element="step", label=f"{self.colors[0]['label']} data"
                    )
                if len(y_state1) > 0:
                    sns.histplot(
                        y_state1, ax=ax, color=self.colors[1]['fill'], 
                        stat="density", kde=False, alpha=0.12, 
                        element="step", label=f"{self.colors[1]['label']} data"
                    )
            
            # 2. Plot Gaussian PDF curves and Vertical Guideline Markers (Mean and +/- 1 Std)
            for state in [0, 1]:
                mu = means[state, d]
                sigma = np.sqrt(np.clip(covariances[state, d, d], a_min=1e-6, a_max=None))
                
                pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
                
                color_config = self.colors[state]
                
                # Plot theoretical normal curve
                ax.plot(
                    x_vals, pdf_vals, color=color_config['line'], 
                    linewidth=2.2, label=f"{color_config['label']} PDF"
                )
                
                # Shading under curve
                ax.fill_between(
                    x_vals, pdf_vals, alpha=0.05, 
                    color=color_config['fill']
                )
                
                # Plot Central Tendency Guideline: Mean (E[X])
                ax.axvline(
                    x=mu, color=color_config['line'], linestyle='--', linewidth=1.5, alpha=0.8,
                    label=None
                )
                
                # Plot Dispersion Guidelines: +/- 1 Std bounds
                ax.axvline(
                    x=mu - sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6,
                    label=None
                )
                ax.axvline(
                    x=mu + sigma, color=color_config['line'], linestyle=':', linewidth=1.0, alpha=0.6,
                    label=None
                )
                
                # Label with symbols mu and sigma next to the lines
                # Import transforms locally if needed, factory created in _plot_row
                y_pos_mean = 0.90 if state == 0 else 0.75
                y_pos_std = 0.83 if state == 0 else 0.68
                
                ax.text(
                    mu, y_pos_mean, f"$\\mu_{state} = {mu:.4f}$", transform=trans, color=color_config['line'],
                    fontsize=8.0, ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
                )
                ax.text(
                    mu + sigma, y_pos_std, f"$\\sigma_{state} = {sigma:.4f}$", transform=trans, color=color_config['line'],
                    fontsize=7.0, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1)
                )
                
            ax.set_title(f"{title_prefix} | Topology: {self.topology_names[d]}", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel("Topology Score", fontsize=9, labelpad=4)
            ax.set_ylabel("Probability Density", fontsize=9, labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # De-duplicate legend entries to keep layout clean and readable
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels[label] = handle
                    
            ax.legend(
                unique_labels.values(), unique_labels.keys(), 
                fontsize=7.5, loc='upper right', framealpha=0.9
            )

    def save_plot(self):
        """Saves generated plot as PNG."""
        if self.phlag.args.output_file:
            output_dir = pathlib.Path(self.phlag.args.output_file).parent
        else:
            repo_root = pathlib.Path(__file__).parent.parent.resolve()
            output_dir = repo_root / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
        suffix = "_ilr" if self.phlag.ilr_transform else ""
        plot_file = output_dir / f"distributions_{self.input_path.stem}{suffix}.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visual distributions plot to: {plot_file}")

def main():
    args = parse_arguments()
    phlag = Phlag(args)
    
    # Ingest baseline setup properties before fitting
    initial_means = np.array(phlag.params.emissions.means)
    initial_covariances = np.array(phlag.params.emissions.covariances)
    
    print("Fitting HMM and running EM to retrieve final distributions...")
    phlag.run()
    
    print("Generating distributions plot...")
    # Executing the object-oriented analysis engine pipeline
    PhlagPlotter(phlag, initial_means, initial_covariances)

if __name__ == "__main__":
    main()
