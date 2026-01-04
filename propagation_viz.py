
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns


OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_propagation_timeline(propagation_results: dict, filename: str = "propagation_timeline.png"):
    """
    Plot how information spreads over time for all categories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'gossip_fake': '#ff6b6b',
        'gossip_real': '#4ecdc4',
        'politi_fake': '#ff8c42',
        'politi_real': '#95e1d3'
    }
    
    labels = {
        'gossip_fake': 'GossipCop Fake',
        'gossip_real': 'GossipCop Real',
        'politi_fake': 'PolitiFact Fake',
        'politi_real': 'PolitiFact Real'
    }
    
    # Independent Cascade Model
    ax1.set_title('Independent Cascade Model - Propagation Over Time', fontsize=14, weight='bold')
    for category, results in propagation_results.items():
        if 'ic_model' in results and results['ic_model'].get('all_results'):
            # Get average timeline across all runs
            all_timelines = [r['timeline'] for r in results['ic_model']['all_results']]
            max_len = max(len(t) for t in all_timelines)
            
            # Pad timelines and average
            padded = []
            for timeline in all_timelines:
                padded_timeline = timeline + [timeline[-1]] * (max_len - len(timeline))
                padded.append(padded_timeline)
            
            avg_timeline = np.mean(padded, axis=0)
            std_timeline = np.std(padded, axis=0)
            
            steps = range(len(avg_timeline))
            ax1.plot(steps, avg_timeline, label=labels[category], 
                    color=colors[category], linewidth=2.5, marker='o', markersize=4)
            ax1.fill_between(steps, 
                           avg_timeline - std_timeline, 
                           avg_timeline + std_timeline,
                           alpha=0.2, color=colors[category])
    
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Number of Activated Nodes', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Linear Threshold Model
    ax2.set_title('Linear Threshold Model - Propagation Over Time', fontsize=14, weight='bold')
    for category, results in propagation_results.items():
        if 'lt_model' in results and results['lt_model'].get('all_results'):
            all_timelines = [r['timeline'] for r in results['lt_model']['all_results']]
            max_len = max(len(t) for t in all_timelines)
            
            padded = []
            for timeline in all_timelines:
                padded_timeline = timeline + [timeline[-1]] * (max_len - len(timeline))
                padded.append(padded_timeline)
            
            avg_timeline = np.mean(padded, axis=0)
            std_timeline = np.std(padded, axis=0)
            
            steps = range(len(avg_timeline))
            ax2.plot(steps, avg_timeline, label=labels[category], 
                    color=colors[category], linewidth=2.5, marker='s', markersize=4)
            ax2.fill_between(steps, 
                           avg_timeline - std_timeline, 
                           avg_timeline + std_timeline,
                           alpha=0.2, color=colors[category])
    
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Number of Activated Nodes', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_activation_rates(propagation_results: dict, filename: str = "activation_rates.png"):
    """
    Compare final activation rates across categories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['GossipCop\nFake', 'GossipCop\nReal', 'PolitiFact\nFake', 'PolitiFact\nReal']
    keys = ['gossip_fake', 'gossip_real', 'politi_fake', 'politi_real']
    colors = ['#ff6b6b', '#4ecdc4', '#ff8c42', '#95e1d3']
    
    # Independent Cascade
    ic_rates = [propagation_results[k]['ic_model']['avg_activation_rate'] * 100 
                for k in keys if 'ic_model' in propagation_results[k]]
    ic_stds = [propagation_results[k]['ic_model']['std_activation_rate'] * 100 
               for k in keys if 'ic_model' in propagation_results[k]]
    
    bars1 = ax1.bar(categories, ic_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.errorbar(categories, ic_rates, yerr=ic_stds, fmt='none', 
                color='black', capsize=5, linewidth=2)
    ax1.set_ylabel('Activation Rate (%)', fontsize=12)
    ax1.set_title('Independent Cascade Model', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(ic_rates) * 1.2)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, ic_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Linear Threshold
    lt_rates = [propagation_results[k]['lt_model']['avg_activation_rate'] * 100 
                for k in keys if 'lt_model' in propagation_results[k]]
    lt_stds = [propagation_results[k]['lt_model']['std_activation_rate'] * 100 
               for k in keys if 'lt_model' in propagation_results[k]]
    
    bars2 = ax2.bar(categories, lt_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.errorbar(categories, lt_rates, yerr=lt_stds, fmt='none', 
                color='black', capsize=5, linewidth=2)
    ax2.set_ylabel('Activation Rate (%)', fontsize=12)
    ax2.set_title('Linear Threshold Model', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(lt_rates) * 1.2 if lt_rates else 1)
    
    for bar, rate in zip(bars2, lt_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_propagation_speed(propagation_results: dict, filename: str = "propagation_speed.png"):
    """
    Compare propagation speed (steps to completion).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    categories = ['GossipCop\nFake', 'GossipCop\nReal', 'PolitiFact\nFake', 'PolitiFact\nReal']
    keys = ['gossip_fake', 'gossip_real', 'politi_fake', 'politi_real']
    
    x = np.arange(len(categories))
    width = 0.35
    
    ic_steps = [propagation_results[k]['ic_model'].get('avg_steps', 0) 
                for k in keys if 'ic_model' in propagation_results[k]]
    ic_step_stds = [propagation_results[k]['ic_model'].get('std_steps', 0) 
                    for k in keys if 'ic_model' in propagation_results[k]]
    
    lt_steps = [propagation_results[k]['lt_model'].get('avg_steps', 0) 
                for k in keys if 'lt_model' in propagation_results[k]]
    lt_step_stds = [propagation_results[k]['lt_model'].get('std_steps', 0) 
                    for k in keys if 'lt_model' in propagation_results[k]]
    
    bars1 = ax.bar(x - width/2, ic_steps, width, label='Independent Cascade',
                   color='#3498db', alpha=0.8, edgecolor='black')
    ax.errorbar(x - width/2, ic_steps, yerr=ic_step_stds, fmt='none', 
               color='black', capsize=4)
    
    bars2 = ax.bar(x + width/2, lt_steps, width, label='Linear Threshold',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.errorbar(x + width/2, lt_steps, yerr=lt_step_stds, fmt='none', 
               color='black', capsize=4)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Average Steps to Completion', fontsize=12)
    ax.set_title('Propagation Speed Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_propagation_heatmap(propagation_results: dict, filename: str = "propagation_heatmap.png"):
    """
    Create heatmap showing propagation characteristics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    categories = ['GossipCop Fake', 'GossipCop Real', 'PolitiFact Fake', 'PolitiFact Real']
    keys = ['gossip_fake', 'gossip_real', 'politi_fake', 'politi_real']
    
    # Prepare data for IC model
    ic_data = []
    for key in keys:
        if 'ic_model' in propagation_results[key]:
            ic_data.append([
                propagation_results[key]['ic_model']['avg_activation_rate'] * 100,
                propagation_results[key]['ic_model'].get('avg_steps', 0),
                propagation_results[key]['ic_model']['std_activation_rate'] * 100
            ])
        else:
            ic_data.append([0, 0, 0])
    
    ic_data = np.array(ic_data)
    
    # Prepare data for LT model
    lt_data = []
    for key in keys:
        if 'lt_model' in propagation_results[key]:
            lt_data.append([
                propagation_results[key]['lt_model']['avg_activation_rate'] * 100,
                propagation_results[key]['lt_model'].get('avg_steps', 0),
                propagation_results[key]['lt_model']['std_activation_rate'] * 100
            ])
        else:
            lt_data.append([0, 0, 0])
    
    lt_data = np.array(lt_data)
    
    metrics = ['Activation\nRate (%)', 'Avg Steps', 'Std Dev (%)']
    
    # IC heatmap
    sns.heatmap(ic_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=metrics, yticklabels=categories, ax=ax1,
                cbar_kws={'label': 'Value'}, linewidths=0.5)
    ax1.set_title('Independent Cascade Model', fontsize=14, weight='bold')
    
    # LT heatmap
    sns.heatmap(lt_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=metrics, yticklabels=categories, ax=ax2,
                cbar_kws={'label': 'Value'}, linewidths=0.5)
    ax2.set_title('Linear Threshold Model', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_fake_vs_real_comparison(propagation_results: dict, 
                                 filename: str = "fake_vs_real_propagation.png"):
    """
    Direct comparison of fake vs real for each domain.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # GossipCop - IC Model
    gf_ic = propagation_results['gossip_fake']['ic_model']['avg_activation_rate'] * 100
    gr_ic = propagation_results['gossip_real']['ic_model']['avg_activation_rate'] * 100
    
    ax1.bar(['Fake', 'Real'], [gf_ic, gr_ic], color=['#ff6b6b', '#4ecdc4'], 
           alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Activation Rate (%)', fontsize=12)
    ax1.set_title('GossipCop - Independent Cascade', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    diff = gf_ic - gr_ic
    ax1.text(0.5, max(gf_ic, gr_ic) * 0.9, 
            f'Difference: {diff:+.1f}%', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # GossipCop - LT Model
    gf_lt = propagation_results['gossip_fake']['lt_model']['avg_activation_rate'] * 100
    gr_lt = propagation_results['gossip_real']['lt_model']['avg_activation_rate'] * 100
    
    ax2.bar(['Fake', 'Real'], [gf_lt, gr_lt], color=['#ff6b6b', '#4ecdc4'], 
           alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Activation Rate (%)', fontsize=12)
    ax2.set_title('GossipCop - Linear Threshold', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    diff = gf_lt - gr_lt
    ax2.text(0.5, max(gf_lt, gr_lt) * 0.9, 
            f'Difference: {diff:+.1f}%', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # PolitiFact - IC Model
    pf_ic = propagation_results['politi_fake']['ic_model']['avg_activation_rate'] * 100
    pr_ic = propagation_results['politi_real']['ic_model']['avg_activation_rate'] * 100
    
    ax3.bar(['Fake', 'Real'], [pf_ic, pr_ic], color=['#ff8c42', '#95e1d3'], 
           alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Activation Rate (%)', fontsize=12)
    ax3.set_title('PolitiFact - Independent Cascade', fontsize=13, weight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    diff = pf_ic - pr_ic
    ax3.text(0.5, max(pf_ic, pr_ic) * 0.9, 
            f'Difference: {diff:+.1f}%', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # PolitiFact - LT Model
    pf_lt = propagation_results['politi_fake']['lt_model']['avg_activation_rate'] * 100
    pr_lt = propagation_results['politi_real']['lt_model']['avg_activation_rate'] * 100
    
    ax4.bar(['Fake', 'Real'], [pf_lt, pr_lt], color=['#ff8c42', '#95e1d3'], 
           alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Activation Rate (%)', fontsize=12)
    ax4.set_title('PolitiFact - Linear Threshold', fontsize=13, weight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    diff = pf_lt - pr_lt
    ax4.text(0.5, max(pf_lt, pr_lt) * 0.9, 
            f'Difference: {diff:+.1f}%', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")