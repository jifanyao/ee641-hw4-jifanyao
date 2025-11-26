"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os
import json


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """
    
    # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    # Arrow characters/symbols for policy visualization
    ACTION_ARROWS = {
        0: u'\u2191',  # Up Arrow
        1: u'\u2192',  # Right Arrow
        2: u'\u2193',  # Down Arrow
        3: u'\u2190'   # Left Arrow
    }

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size

        # Define special positions (row, col)
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]
        
        # Mapping (row, col) to special cell type
        self.special_cells = {
            self.start_pos: ('S', 'blue'),
            self.goal_pos: ('G', 'green'),
        }
        for pos in self.obstacles:
            self.special_cells[pos] = ('X', 'black')
        for pos in self.penalties:
            self.special_cells[pos] = ('P', 'red')

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to grid position (row, col)."""
        return (state // self.grid_size, state % self.grid_size)

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function", filename: str = "value_function.png") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
            filename: Name of the file to save
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Reshape values to 2D grid
        value_grid = values.reshape((self.grid_size, self.grid_size))
        
        # Set color scale (use 0.0 for terminal state goal)
        vmin = np.min(values)
        vmax = np.max(values)
        
        im = ax.imshow(value_grid, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.set_xticklabels(np.arange(self.grid_size))
        ax.set_yticklabels(np.arange(self.grid_size))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Mark special cells and overlay value text
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                
                # Overlay value text
                value_str = f'{value_grid[r, c]:.2f}'
                text_color = "w" if value_grid[r, c] < (vmax + vmin) / 2 else "k" # Choose text color based on background
                ax.text(c, r, value_str,
                        ha="center", va="center", color=text_color,
                        fontsize=8, fontweight='bold')
                
                # Mark special cells with border
                if pos in self.special_cells:
                    label, color = self.special_cells[pos]
                    rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, 
                                         edgecolor=color, facecolor='none', linewidth=3)
                    ax.add_patch(rect)
                    # For obstacles, might be better to just show the border
                    if pos != self.goal_pos and pos != self.start_pos: # Add S/P/X/G label clearly
                        ax.text(c, r, label, ha="center", va="center", color='k', 
                                fontsize=10, fontweight='extra bold', bbox=dict(facecolor=color, alpha=0.3, edgecolor='none', boxstyle='square,pad=0.2'))


        # Add colorbar and labels
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('State Value V(s)')
        
        # Add title
        ax.set_title(title)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join('results/visualizations', filename))
        plt.close(fig)

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy", filename: str = "policy.png") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
            filename: Name of the file to save
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create grid plot (white background, grid lines)
        grid_data = np.zeros((self.grid_size, self.grid_size))
        im = ax.imshow(grid_data, cmap='binary', origin='upper', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.set_xticklabels(np.arange(self.grid_size))
        ax.set_yticklabels(np.arange(self.grid_size))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.grid(which='major', color='gray', linestyle='-', linewidth=1)
        
        # For each state:
        for s in range(self.n_states):
            r, c = self._state_to_pos(s)
            pos = (r, c)
            
            # Mark start, goal, obstacles, penalties (with solid color)
            facecolor = 'white'
            label = ''
            
            if pos == self.start_pos:
                facecolor = 'lightblue'
                label = 'S'
            elif pos == self.goal_pos:
                facecolor = 'lightgreen'
                label = 'G'
            elif pos in self.obstacles:
                facecolor = 'darkgray'
                label = 'X'
            elif pos in self.penalties:
                facecolor = 'lightcoral'
                label = 'P'

            # Draw cell background/border
            rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, 
                                 edgecolor='gray', facecolor=facecolor, linewidth=1)
            ax.add_patch(rect)
            
            # Draw label
            if label:
                 ax.text(c, r, label, ha="center", va="center", color='black', 
                         fontsize=12, fontweight='bold')
                         
            # Draw arrow indicating action direction
            # Policy is meaningful only for non-terminal, non-obstacle states
            if not (pos == self.goal_pos or pos in self.obstacles):
                action = policy[s]
                arrow = self.ACTION_ARROWS.get(action, '')
                
                # Text is used for simpler arrow rendering in grid cells
                ax.text(c, r, arrow, 
                        ha="center", va="center", fontsize=16, 
                        color='black', fontweight='bold')


        ax.set_title(title)
        ax.tick_params(which='both', length=0) # Hide ticks
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join('results/visualizations', filename))
        plt.close(fig)

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function", filename: str = "q_function.png") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
            filename: Name of the file to save
        """
        n_actions = q_values.shape[1]
        fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 5), squeeze=False)
        axes = axes.flatten()
        
        # Find global min/max for consistent colormap
        vmin = np.min(q_values)
        vmax = np.max(q_values)
        
        for a in range(n_actions):
            ax = axes[a]
            action_name = ['UP', 'RIGHT', 'DOWN', 'LEFT'][a]
            
            # Show Q-values as heatmap
            q_grid = q_values[:, a].reshape((self.grid_size, self.grid_size))
            im = ax.imshow(q_grid, cmap='coolwarm', origin='upper', vmin=vmin, vmax=vmax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(self.grid_size))
            ax.set_yticks(np.arange(self.grid_size))
            ax.set_xticklabels(np.arange(self.grid_size))
            ax.set_yticklabels(np.arange(self.grid_size))
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_title(f"Action: {action_name}")
            
            # Mark special cells and overlay Q-value text
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    pos = (r, c)
                    
                    # Overlay Q-value text
                    value_str = f'{q_grid[r, c]:.2f}'
                    text_color = "w" if q_grid[r, c] < (vmax + vmin) / 2 else "k"
                    ax.text(c, r, value_str,
                            ha="center", va="center", color=text_color,
                            fontsize=8, fontweight='bold')

                    # Mark special cells with border
                    if pos in self.special_cells:
                        _, color = self.special_cells[pos]
                        rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, 
                                             edgecolor=color, facecolor='none', linewidth=3)
                        ax.add_patch(rect)


        # Add overall title and colorbar
        fig.suptitle(title, fontsize=16)
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.03, pad=0.04).set_label('Q-Value Q(s,a)')
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join('results/visualizations', filename))
        plt.close(fig)

    def plot_convergence(self, vi_history: list, qi_history: list, filename: str = "convergence_plot.png") -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history (max Bellman error)
            qi_history: Q-iteration convergence history (max Bellman error)
            filename: Name of the file to save
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Bellman error vs iteration for both algorithms
        ax.plot(range(1, len(vi_history) + 1), vi_history, label='Value Iteration ($||V_{k+1}-V_{k}||_{\infty}$)', marker='.', linestyle='-')
        ax.plot(range(1, len(qi_history) + 1), qi_history, label='Q-Iteration ($||Q_{k+1}-Q_{k}||_{\infty}$)', marker='.', linestyle='-')
        
        # Use log scale for y-axis
        ax.set_yscale('log')
        
        # Add legend and labels
        ax.set_xlabel('Iteration Number (k)')
        ax.set_ylabel('Max Bellman Error (log scale)')
        ax.set_title('Convergence of Value Iteration and Q-Iteration')
        ax.legend()
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join('results/visualizations', filename))
        plt.close(fig)

    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray, 
                                vi_iterations: int, qi_iterations: int, filename: str = "comparison_figure.png") -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
            vi_iterations: Iterations for VI
            qi_iterations: Iterations for QI
            filename: Name of the file to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Value Iteration vs Q-Iteration Comparison', fontsize=16)
        
        # Global value scale
        all_values = np.concatenate([vi_values, qi_values])
        vmin = np.min(all_values)
        vmax = np.max(all_values)
        
        # --- Plotting Helper for Value Function (Heatmap) ---
        def plot_val(ax, values, title):
            value_grid = values.reshape((self.grid_size, self.grid_size))
            im = ax.imshow(value_grid, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(self.grid_size))
            ax.set_yticks(np.arange(self.grid_size))
            ax.set_xticklabels(np.arange(self.grid_size))
            ax.set_yticklabels(np.arange(self.grid_size))
            
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    pos = (r, c)
                    value_str = f'{value_grid[r, c]:.2f}'
                    text_color = "w" if value_grid[r, c] < (vmax + vmin) / 2 else "k"
                    ax.text(c, r, value_str,
                            ha="center", va="center", color=text_color,
                            fontsize=8, fontweight='bold')
                    if pos in self.special_cells:
                        _, color = self.special_cells[pos]
                        rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, 
                                             edgecolor=color, facecolor='none', linewidth=3)
                        ax.add_patch(rect)
            ax.set_title(title)
            return im

        # --- Plotting Helper for Policy (Arrows) ---
        def plot_pol(ax, policy, title, other_policy=None):
            grid_data = np.zeros((self.grid_size, self.grid_size))
            ax.imshow(grid_data, cmap='binary', origin='upper', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(self.grid_size))
            ax.set_yticks(np.arange(self.grid_size))
            ax.set_xticklabels(np.arange(self.grid_size))
            ax.set_yticklabels(np.arange(self.grid_size))
            ax.grid(which='major', color='gray', linestyle='-', linewidth=1)
            ax.tick_params(which='both', length=0)

            for s in range(self.n_states):
                r, c = self._state_to_pos(s)
                pos = (r, c)
                
                facecolor = 'white'
                label = ''
                if pos == self.start_pos: facecolor, label = 'lightblue', 'S'
                elif pos == self.goal_pos: facecolor, label = 'lightgreen', 'G'
                elif pos in self.obstacles: facecolor, label = 'darkgray', 'X'
                elif pos in self.penalties: facecolor, label = 'lightcoral', 'P'

                # Draw cell background
                rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, 
                                     edgecolor='gray', facecolor=facecolor, linewidth=1)
                ax.add_patch(rect)
                if label:
                     ax.text(c, r, label, ha="center", va="center", color='black', 
                             fontsize=12, fontweight='bold')
                             
                if not (pos == self.goal_pos or pos in self.obstacles):
                    action = policy[s]
                    arrow = self.ACTION_ARROWS.get(action, '')
                    
                    # Highlight any differences
                    arrow_color = 'black'
                    if other_policy is not None and not np.array_equal(policy, other_policy) and policy[s] != other_policy[s]:
                        arrow_color = 'magenta' # Highlighted difference
                        
                    ax.text(c, r, arrow, 
                            ha="center", va="center", fontsize=16, 
                            color=arrow_color, fontweight='bold')
            ax.set_title(title)

        
        # Top left: VI value function
        im = plot_val(axes[0, 0], vi_values, f"VI Value Function ($V^*$) - {vi_iterations} iters")
        # Top right: QI value function
        im_qi = plot_val(axes[0, 1], qi_values, f"QI Value Function ($V^*$) - {qi_iterations} iters")
        # Add a single colorbar for value functions
        fig.colorbar(im, ax=[axes[0, 0], axes[0, 1]], orientation='vertical', fraction=0.046, pad=0.04).set_label('State Value $V(s)$')
        
        # Bottom left: VI policy
        plot_pol(axes[1, 0], vi_policy, "VI Optimal Policy ($\pi^* = \operatorname{argmax}_a Q(s,a)$)", qi_policy)
        # Bottom right: QI policy
        plot_pol(axes[1, 1], qi_policy, "QI Optimal Policy ($\pi^* = \operatorname{argmax}_a Q(s,a)$)", vi_policy)
        
        # Save comprehensive comparison figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join('results/visualizations', filename))
        plt.close(fig)


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    print("\n--- Visualizing Results ---")
    
    # Load saved value functions and policies
    try:
        with open('results/vi_results.json', 'r') as f:
            vi_results = json.load(f)
        with open('results/qi_results.json', 'r') as f:
            qi_results = json.load(f)
        with open('results/comparison_results.json', 'r') as f:
            comp_results = json.load(f)
            
    except FileNotFoundError:
        print("Error: Results files not found. Run 'train.py' first.")
        return

    # Convert lists back to NumPy arrays
    vi_values = np.array(vi_results['values'])
    vi_policy = np.array(vi_results['policy'], dtype=np.int32)
    vi_history = vi_results['error_history']
    
    qi_q_values = np.array(qi_results['q_values'])
    qi_values = np.array(qi_results['values'])
    qi_policy = np.array(qi_results['policy'], dtype=np.int32)
    qi_history = qi_results['error_history']

    # Create visualizer instance
    visualizer = GridWorldVisualizer()
    
    # Generate all visualization plots
    print("Generating visualizations...")
    visualizer.plot_value_function(vi_values, title="Value Iteration Optimal Value Function", filename="vi_value_function.png")
    visualizer.plot_policy(vi_policy, title="Value Iteration Optimal Policy", filename="vi_policy.png")
    
    visualizer.plot_value_function(qi_values, title="Q-Iteration Optimal Value Function", filename="qi_value_function.png")
    visualizer.plot_policy(qi_policy, title="Q-Iteration Optimal Policy", filename="qi_policy.png")
    visualizer.plot_q_function(qi_q_values, title="Q-Iteration Optimal Q-Function", filename="qi_q_function.png")
    
    visualizer.plot_convergence(vi_history, qi_history, filename="convergence_plot.png")
    
    visualizer.create_comparison_figure(
        vi_values, qi_values, vi_policy, qi_policy, 
        vi_results['iterations'], qi_results['iterations'],
        filename="comparison_figure.png"
    )
    
    print("All visualization plots saved to results/visualizations/")
    
    # Print summary statistics (as required by report)
    print("\n--- Summary Statistics (for Report) ---")
    print(f"1. Iterations until convergence:")
    print(f"   - Value Iteration: {vi_results['iterations']}")
    print(f"   - Q-Iteration:     {qi_results['iterations']}")
    
    print(f"4. Policy Match: {comp_results['policies_match']}")
    print(f"   Max Value Diff: {comp_results['max_value_diff']:.6e}")
    print(f"Comparison plots (Value function and policy) are saved as comparison_figure.png.")


if __name__ == '__main__':
    visualize_results()