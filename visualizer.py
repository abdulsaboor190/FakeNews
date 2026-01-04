import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_degree_distribution(G: nx.Graph, title: str, filename: str) -> None:
    """Plot degree distribution for the graph."""
    degrees = [G.degree(n) for n in G.nodes()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title} - Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot (power law check)
    degree_count = {}
    for d in degrees:
        degree_count[d] = degree_count.get(d, 0) + 1
    
    degrees_list = sorted(degree_count.keys())
    counts = [degree_count[d] for d in degrees_list]
    
    ax2.loglog(degrees_list, counts, 'bo', alpha=0.6)
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title(f'{title} - Log-Log Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_network_sample(G: nx.Graph, title: str, filename: str, max_nodes: int = 200) -> None:
    """Plot a sample of the network (for visualization purposes)."""
    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        # Get largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(list(largest_cc)[:max_nodes])
    else:
        subgraph = G
    
    plt.figure(figsize=(12, 10))
    
    # Color nodes by bipartite type
    news_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("bipartite") == "news"]
    user_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("bipartite") == "user"]
    
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    
    # Draw news nodes in red
    nx.draw_networkx_nodes(subgraph, pos, nodelist=news_nodes, 
                           node_color='red', node_size=100, 
                           alpha=0.7, label='News')
    
    # Draw user nodes in blue
    nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes, 
                           node_color='blue', node_size=50, 
                           alpha=0.5, label='Users')
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5)
    
    plt.title(f'{title} - Network Sample (n={subgraph.number_of_nodes()})')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_metrics_comparison(metrics_dict: dict, filename: str = "metrics_comparison.png") -> None:
    """
    Create bar charts comparing metrics across all four categories.
    
    metrics_dict format: {
        'gossip_fake': {...metrics...},
        'gossip_real': {...metrics...},
        'politi_fake': {...metrics...},
        'politi_real': {...metrics...}
    }
    """
    categories = ['GossipCop\nFake', 'GossipCop\nReal', 'PolitiFact\nFake', 'PolitiFact\nReal']
    keys = ['gossip_fake', 'gossip_real', 'politi_fake', 'politi_real']
    
    # Select important metrics to compare
    metrics_to_plot = [
        ('avg_engagement_per_article', 'Avg Engagement per Article'),
        ('news_avg_degree', 'Avg News Degree'),
        ('user_avg_degree', 'Avg User Degree'),
        ('density', 'Network Density'),
        ('news_clustering', 'News Clustering Coefficient'),
        ('news_avg_centrality', 'Avg News Centrality')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [metrics_dict[k].get(metric_key, 0) for k in keys]
        
        colors = ['#ff6b6b', '#4ecdc4', '#ff6b6b', '#4ecdc4']  # Red for fake, teal for real
        
        axes[idx].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric_label)
        axes[idx].set_title(metric_label)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_engagement_distribution(all_graphs: dict, filename: str = "engagement_distribution.png") -> None:
    """
    Plot engagement distribution (news article degree) for all four categories.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    titles = {
        'gossip_fake': 'GossipCop Fake',
        'gossip_real': 'GossipCop Real',
        'politi_fake': 'PolitiFact Fake',
        'politi_real': 'PolitiFact Real'
    }
    
    for idx, (key, (G, metrics)) in enumerate(all_graphs.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Get news node degrees
        news_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "news"]
        news_degrees = [G.degree(n) for n in news_nodes]
        
        ax.hist(news_degrees, bins=30, edgecolor='black', alpha=0.7, 
                color='#ff6b6b' if 'fake' in key else '#4ecdc4')
        ax.set_xlabel('Number of Engagements (Degree)')
        ax.set_ylabel('Number of Articles')
        ax.set_title(titles[key])
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_deg = np.mean(news_degrees)
        median_deg = np.median(news_degrees)
        ax.text(0.65, 0.95, f'Mean: {mean_deg:.1f}\nMedian: {median_deg:.1f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_summary_table(metrics_dict: dict, filename: str = "summary_table.png") -> None:
    """Create a table summarizing key metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    rows = []
    row_labels = [
        'Total Nodes',
        'News Articles',
        'Unique Users',
        'Total Edges',
        'Avg Engagement/Article',
        'Network Density',
        'Avg News Degree',
        'News Clustering',
        'Avg Centrality'
    ]
    
    metric_keys = [
        'total_nodes',
        'news_articles',
        'unique_users',
        'total_edges',
        'avg_engagement_per_article',
        'density',
        'news_avg_degree',
        'news_clustering',
        'news_avg_centrality'
    ]
    
    categories = ['GossipCop\nFake', 'GossipCop\nReal', 'PolitiFact\nFake', 'PolitiFact\nReal']
    keys = ['gossip_fake', 'gossip_real', 'politi_fake', 'politi_real']
    
    table_data = []
    for metric_key in metric_keys:
        row = [metrics_dict[k].get(metric_key, 0) for k in keys]
        table_data.append([f'{v:.4f}' if isinstance(v, float) else str(v) for v in row])
    
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=categories,
                     cellLoc='center', loc='center', colWidths=[0.2]*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(categories)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color row labels
    for i in range(len(row_labels)):
        table[(i+1, -1)].set_facecolor('#E7E6E6')
    
    plt.title('Fake News Propagation Analysis - Summary Metrics', 
              fontsize=14, weight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")