import sys
import pathlib
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

from phlag.main import Phlag, parse_arguments

def generate_distributions_plot(phlag, initial_means, initial_covariances):
    """
    Plots the empirical histograms and fitted Gaussian PDF curves
    for each topology in both states (State 0: Null, State 1: Alternative)
    both before and after EM fitting.
    """
    # Calculate most likely states (Viterbi assignment)
    most_likely_states = phlag.hmm.most_likely_states(phlag.params, phlag.Y)
    
    # Extract fitted emission parameters (after EM)
    final_means = np.array(phlag.params.emissions.means)
    final_covariances = np.array(phlag.params.emissions.covariances)
    
    emission_dim = phlag.Y.shape[-1]
    
    # Setup premium styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'axes.edgecolor': '#cccccc',
        'grid.color': '#f0f0f0'
    })
    
    # Define premium colors
    # State 0 (Null/Background): Deep Steel Blue
    # State 1 (Alternative/Anomalous): Warm Coral
    colors = {
        0: {
            'line': '#2B4C7E',
            'fill': '#2B4C7E',
            'label': 'State 0 (Null / Background)'
        },
        1: {
            'line': '#E05A47',
            'fill': '#E05A47',
            'label': 'State 1 (Alternative / Anomalous)'
        }
    }
    
    # 2 rows (Before EM and After EM), emission_dim columns
    fig, axes = plt.subplots(2, emission_dim, figsize=(5 * emission_dim, 8.5), sharey=False)
    
    # Ensure axes is a 2D array of shape (2, emission_dim) for easy indexing
    if emission_dim == 1:
        axes = np.array([[axes[0]], [axes[1]]])
        
    # Map coordinates to topology names if we have the standard 3 topologies
    if not phlag.ilr_transform and emission_dim == 3:
        topology_names = ["ABBA", "BABA", "AABB"]
    else:
        topology_names = [f"ILR Coordinate {i+1}" if phlag.ilr_transform else f"Coordinate {i+1}" for i in range(emission_dim)]
        
    # Find min/max data range to plot curves nicely across both rows
    ranges = {}
    for d in range(emission_dim):
        ymin, ymax = float(np.min(phlag.Y[:, d])), float(np.max(phlag.Y[:, d]))
        ypad = (ymax - ymin) * 0.15 or 0.1
        ranges[d] = np.linspace(ymin - ypad, ymax + ypad, 300)

    # --- ROW 0: BEFORE EM (Initial Distributions) ---
    for d in range(emission_dim):
        ax = axes[0, d]
        x_vals = ranges[d]
        
        for state in [0, 1]:
            mu = initial_means[state, d]
            sigma = np.sqrt(np.clip(initial_covariances[state, d, d], a_min=1e-6, a_max=None))
            
            pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
            
            # Plot the line
            ax.plot(
                x_vals, pdf_vals, color=colors[state]['line'], 
                linewidth=2.5, label=f"{colors[state]['label']} PDF"
            )
            # Fill under the curve
            ax.fill_between(
                x_vals, pdf_vals, alpha=0.08, 
                color=colors[state]['fill']
            )
            
        ax.set_title(f"Before EM | Topology: {topology_names[d]}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Topology Score", fontsize=10, labelpad=5)
        ax.set_ylabel("Probability Density", fontsize=10, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')

    # --- ROW 1: AFTER EM (Fitted Distributions & Data Histograms) ---
    for d in range(emission_dim):
        ax = axes[1, d]
        x_vals = ranges[d]
        
        # 1. Plot Empirical Histograms
        y_state0 = np.array(phlag.Y[most_likely_states == 0, d])
        y_state1 = np.array(phlag.Y[most_likely_states == 1, d])
        
        if len(y_state0) > 0:
            sns.histplot(
                y_state0, ax=ax, color=colors[0]['fill'], 
                stat="density", kde=False, alpha=0.15, 
                element="step", label=f"{colors[0]['label']} data"
            )
        if len(y_state1) > 0:
            sns.histplot(
                y_state1, ax=ax, color=colors[1]['fill'], 
                stat="density", kde=False, alpha=0.15, 
                element="step", label=f"{colors[1]['label']} data"
            )
            
        # 2. Plot Theoretical Fitted Gaussian PDFs
        for state in [0, 1]:
            mu = final_means[state, d]
            sigma = np.sqrt(np.clip(final_covariances[state, d, d], a_min=1e-6, a_max=None))
            
            pdf_vals = stats.norm.pdf(x_vals, mu, sigma)
            
            # Plot the line
            ax.plot(
                x_vals, pdf_vals, color=colors[state]['line'], 
                linewidth=2.5, label=f"{colors[state]['label']} PDF"
            )
            # Fill under the curve
            ax.fill_between(
                x_vals, pdf_vals, alpha=0.08, 
                color=colors[state]['fill']
            )
            
        ax.set_title(f"After EM | Topology: {topology_names[d]}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Topology Score", fontsize=10, labelpad=5)
        ax.set_ylabel("Probability Density", fontsize=10, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        
    plt.tight_layout()
    
    # Save the plot
    input_path = pathlib.Path(phlag.args.caster_scores)
    
    if phlag.args.output_file:
        output_dir = pathlib.Path(phlag.args.output_file).parent
    else:
        # Default to test/ directory or current directory
        repo_root = pathlib.Path(__file__).parent.parent.resolve()
        output_dir = repo_root / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
    suffix = "_ilr" if phlag.ilr_transform else ""
    plot_file = output_dir / f"distributions_{input_path.stem}{suffix}.png"
    
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visual distributions plot to: {plot_file}")

def main():
    args = parse_arguments()
    phlag = Phlag(args)
    
    # Capture initial parameters before running EM
    initial_means = np.array(phlag.params.emissions.means)
    initial_covariances = np.array(phlag.params.emissions.covariances)
    
    print("Fitting HMM and running EM to retrieve final distributions...")
    phlag.run()
    
    print("Generating distributions plot...")
    generate_distributions_plot(phlag, initial_means, initial_covariances)

if __name__ == "__main__":
    main()
