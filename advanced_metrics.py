
import networkx as nx
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


def calculate_centrality_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Calculate various centrality measures to identify influential nodes.
    """
    metrics = {}
    
    # Skip if graph is too small
    if G.number_of_nodes() < 2:
        return {
            'avg_degree_centrality': 0,
            'avg_betweenness_centrality': 0,
            'avg_closeness_centrality': 0,
            'max_degree_centrality': 0,
            'centrality_variance': 0
        }
    
    # Degree centrality (fast)
    degree_cent = nx.degree_centrality(G)
    metrics['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
    metrics['max_degree_centrality'] = np.max(list(degree_cent.values()))
    metrics['centrality_variance'] = np.var(list(degree_cent.values()))
    
    # Betweenness centrality (expensive, sample for large graphs)
    if G.number_of_nodes() < 500:
        between_cent = nx.betweenness_centrality(G)
    else:
        between_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    metrics['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
    metrics['max_betweenness_centrality'] = np.max(list(between_cent.values()))
    
    # Closeness centrality (only for connected components)
    if nx.is_connected(G):
        close_cent = nx.closeness_centrality(G)
        metrics['avg_closeness_centrality'] = np.mean(list(close_cent.values()))
    else:
        # Calculate for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        close_cent = nx.closeness_centrality(subgraph)
        metrics['avg_closeness_centrality'] = np.mean(list(close_cent.values()))
    
    return metrics


def analyze_community_structure(G: nx.Graph) -> Dict[str, float]:
    """
    Analyze community structure using various algorithms.
    """
    metrics = {}
    
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return {
            'num_communities': 0,
            'modularity': 0,
            'avg_community_size': 0
        }
    
    try:
        # Louvain community detection (fast and effective)
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        
        # Number of communities
        num_communities = len(set(communities.values()))
        metrics['num_communities'] = num_communities
        
        # Modularity
        modularity = community_louvain.modularity(communities, G)
        metrics['modularity'] = modularity
        
        # Average community size
        community_sizes = {}
        for node, comm_id in communities.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        metrics['avg_community_size'] = np.mean(list(community_sizes.values()))
        metrics['max_community_size'] = np.max(list(community_sizes.values()))
        
    except ImportError:
        # Fallback: use greedy modularity communities
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        metrics['num_communities'] = len(communities)
        metrics['avg_community_size'] = np.mean([len(c) for c in communities])
        metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
    
    return metrics


def calculate_assortativity(G: nx.Graph) -> Dict[str, float]:
    """
    Calculate assortativity (tendency of similar nodes to connect).
    """
    metrics = {}
    
    if G.number_of_edges() == 0:
        return {'degree_assortativity': 0}
    
    try:
        # Degree assortativity
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        metrics['degree_assortativity'] = 0
    
    return metrics


def analyze_path_lengths(G: nx.Graph) -> Dict[str, float]:
    """
    Analyze path length characteristics (how quickly info can spread).
    """
    metrics = {}
    
    if G.number_of_nodes() < 2:
        return {
            'avg_shortest_path': 0,
            'diameter': 0,
            'radius': 0,
            'efficiency': 0
        }
    
    # Work with largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    if G.number_of_nodes() < 2:
        return {
            'avg_shortest_path': 0,
            'diameter': 0,
            'radius': 0,
            'efficiency': 0
        }
    
    try:
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
        metrics['radius'] = nx.radius(G)
        metrics['efficiency'] = nx.global_efficiency(G)
    except:
        # For very large graphs, sample
        metrics['avg_shortest_path'] = 0
        metrics['diameter'] = 0
        metrics['radius'] = 0
        metrics['efficiency'] = nx.global_efficiency(G)
    
    return metrics


def calculate_robustness_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Measure network robustness (resilience to node removal).
    """
    metrics = {}
    
    if G.number_of_nodes() == 0:
        return {'robustness_score': 0}
    
    # Node connectivity (minimum nodes to remove to disconnect)
    try:
        if nx.is_connected(G):
            metrics['node_connectivity'] = nx.node_connectivity(G)
        else:
            metrics['node_connectivity'] = 0
    except:
        metrics['node_connectivity'] = 0
    
    # Edge connectivity
    try:
        if nx.is_connected(G):
            metrics['edge_connectivity'] = nx.edge_connectivity(G)
        else:
            metrics['edge_connectivity'] = 0
    except:
        metrics['edge_connectivity'] = 0
    
    return metrics


def analyze_small_world_properties(G: nx.Graph) -> Dict[str, float]:
    """
    Check if network exhibits small-world properties.
    Small-world: high clustering + short path lengths.
    """
    metrics = {}
    
    # Very small or extremely large graphs: skip expensive small-world computation
    if G.number_of_nodes() < 10 or G.number_of_edges() == 0 or G.number_of_nodes() > 2000:
        return {
            'small_world_sigma': 0,
            'small_world_omega': 0
        }
    
    try:
        # Small-world coefficient (sigma and omega)
        # Note: This can be slow for large graphs
        if G.number_of_nodes() < 1000:
            metrics['small_world_sigma'] = nx.sigma(G, niter=5, nrand=2)
            metrics['small_world_omega'] = nx.omega(G, niter=5, nrand=2)
        else:
            metrics['small_world_sigma'] = 0
            metrics['small_world_omega'] = 0
    except:
        metrics['small_world_sigma'] = 0
        metrics['small_world_omega'] = 0
    
    return metrics

def analyze_graph_coloring(G: nx.Graph) -> Dict[str, float]:
    """
    Analyze graph coloring properties to identify independent sets
    and clustering patterns in news propagation networks.
    
    Graph coloring reveals how news articles can be partitioned into
    groups that don't share audiences.
    """
    metrics = {}
    
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {
            'chromatic_number_estimate': 0,
            'num_independent_sets': 0,
            'avg_independent_set_size': 0,
            'coloring_efficiency': 0
        }
    
    try:
        # For smaller graphs, use greedy coloring
        if G.number_of_nodes() < 1000:
            coloring = nx.greedy_color(G, strategy='largest_first')
            num_colors = max(coloring.values()) + 1
            metrics['chromatic_number_estimate'] = num_colors
            
            # Analyze color classes (independent sets)
            color_classes = {}
            for node, color in coloring.items():
                if color not in color_classes:
                    color_classes[color] = []
                color_classes[color].append(node)
            
            metrics['num_independent_sets'] = len(color_classes)
            metrics['avg_independent_set_size'] = np.mean([len(c) for c in color_classes.values()])
            
            # Coloring efficiency: lower is better (fewer colors needed)
            max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
            metrics['coloring_efficiency'] = num_colors / (max_degree + 1)
            
            # Find maximum independent set size (articles with no shared users)
            metrics['max_independent_set_size'] = max(len(c) for c in color_classes.values())
            
        else:
            # For large graphs, estimate using degree
            max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
            metrics['chromatic_number_estimate'] = max_degree + 1
            metrics['num_independent_sets'] = 0
            metrics['avg_independent_set_size'] = 0
            metrics['coloring_efficiency'] = 0
    
    except Exception as e:
        metrics['chromatic_number_estimate'] = 0
        metrics['num_independent_sets'] = 0
        metrics['avg_independent_set_size'] = 0
        metrics['coloring_efficiency'] = 0
    
    return metrics

def calculate_graph_similarity(G1: nx.Graph, G2: nx.Graph, 
                               label1: str = "Graph 1", 
                               label2: str = "Graph 2") -> Dict[str, float]:
    """
    Calculate structural similarity between two graphs using graph theory metrics.
    This helps quantify how similar fake vs real news networks are structurally.
    """
    metrics = {}
    
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return {
            'spectral_distance': 0,
            'degree_sequence_correlation': 0,
            'clustering_difference': 0,
            'density_difference': 0
        }
    
    try:
        # 1. Spectral similarity (eigenvalue comparison)
        # Laplacian spectrum captures global structure
        if G1.number_of_nodes() > 1 and G2.number_of_nodes() > 1:
            laplacian1 = nx.laplacian_spectrum(G1)
            laplacian2 = nx.laplacian_spectrum(G2)
            
            # Normalize and compare first k eigenvalues
            k = min(10, len(laplacian1), len(laplacian2))
            spectrum1_norm = laplacian1[:k] / np.max(laplacian1[:k]) if np.max(laplacian1[:k]) > 0 else laplacian1[:k]
            spectrum2_norm = laplacian2[:k] / np.max(laplacian2[:k]) if np.max(laplacian2[:k]) > 0 else laplacian2[:k]
            
            metrics['spectral_distance'] = np.linalg.norm(spectrum1_norm - spectrum2_norm)
        else:
            metrics['spectral_distance'] = 0
        
        # 2. Degree sequence similarity
        deg_seq1 = sorted([d for n, d in G1.degree()], reverse=True)
        deg_seq2 = sorted([d for n, d in G2.degree()], reverse=True)
        
        min_len = min(len(deg_seq1), len(deg_seq2))
        if min_len > 1:
            metrics['degree_sequence_correlation'] = np.corrcoef(
                deg_seq1[:min_len],
                deg_seq2[:min_len]
            )[0, 1]
        else:
            metrics['degree_sequence_correlation'] = 0
        
        # 3. Clustering coefficient difference
        clustering1 = nx.average_clustering(G1)
        clustering2 = nx.average_clustering(G2)
        metrics['clustering_difference'] = abs(clustering1 - clustering2)
        
        # 4. Density difference
        density1 = nx.density(G1)
        density2 = nx.density(G2)
        metrics['density_difference'] = abs(density1 - density2)
        
        # 5. Degree distribution similarity (KS test already done elsewhere, but add moment comparison)
        if deg_seq1 and deg_seq2:
            # Compare first three moments
            mean1, mean2 = np.mean(deg_seq1), np.mean(deg_seq2)
            std1, std2 = np.std(deg_seq1), np.std(deg_seq2)
            skew1 = np.mean([(x - mean1)**3 for x in deg_seq1]) / (std1**3) if std1 > 0 else 0
            skew2 = np.mean([(x - mean2)**3 for x in deg_seq2]) / (std2**3) if std2 > 0 else 0
            
            metrics['mean_degree_difference'] = abs(mean1 - mean2)
            metrics['std_degree_difference'] = abs(std1 - std2)
            metrics['skewness_difference'] = abs(skew1 - skew2)
        
        # 6. Structural similarity score (composite metric)
        # Lower score = more similar
        similarity_score = (
            metrics.get('spectral_distance', 0) * 0.3 +
            (1 - abs(metrics.get('degree_sequence_correlation', 0))) * 0.3 +
            metrics.get('clustering_difference', 0) * 0.2 +
            metrics.get('density_difference', 0) * 0.2
        )
        metrics['structural_similarity_score'] = similarity_score
        metrics['graphs_compared'] = f"{label1} vs {label2}"
        
    except Exception as e:
        print(f"Error in graph similarity calculation: {e}")
        metrics['spectral_distance'] = 0
        metrics['degree_sequence_correlation'] = 0
        metrics['structural_similarity_score'] = 0
    
    return metrics

def run_graph_similarity_analysis(graphs_dict: Dict[str, Tuple[nx.Graph, dict]]) -> pd.DataFrame:
    """
    Run graph similarity analysis comparing fake vs real news networks.
    """
    print("\n  Running graph similarity analysis...")
    
    results = []
    
    # Compare GossipCop: Fake vs Real
    G_gf, _ = graphs_dict['gossip_fake']
    G_gr, _ = graphs_dict['gossip_real']
    
    gossip_similarity = calculate_graph_similarity(
        G_gf, G_gr, "GossipCop_Fake", "GossipCop_Real"
    )
    gossip_similarity['domain'] = 'GossipCop'
    gossip_similarity['comparison'] = 'Fake vs Real'
    results.append(gossip_similarity)
    
    # Compare PolitiFact: Fake vs Real
    G_pf, _ = graphs_dict['politi_fake']
    G_pr, _ = graphs_dict['politi_real']
    
    politi_similarity = calculate_graph_similarity(
        G_pf, G_pr, "PolitiFact_Fake", "PolitiFact_Real"
    )
    politi_similarity['domain'] = 'PolitiFact'
    politi_similarity['comparison'] = 'Fake vs Real'
    results.append(politi_similarity)
    
    # Cross-domain comparisons
    # GossipCop Fake vs PolitiFact Fake
    cross_fake = calculate_graph_similarity(
        G_gf, G_pf, "GossipCop_Fake", "PolitiFact_Fake"
    )
    cross_fake['domain'] = 'Cross-Domain'
    cross_fake['comparison'] = 'Fake: GossipCop vs PolitiFact'
    results.append(cross_fake)
    
    # GossipCop Real vs PolitiFact Real
    cross_real = calculate_graph_similarity(
        G_gr, G_pr, "GossipCop_Real", "PolitiFact_Real"
    )
    cross_real['domain'] = 'Cross-Domain'
    cross_real['comparison'] = 'Real: GossipCop vs PolitiFact'
    results.append(cross_real)
    
    return pd.DataFrame(results)

def compare_degree_distributions(G1: nx.Graph, G2: nx.Graph, 
                                 label1: str = "Graph 1", 
                                 label2: str = "Graph 2") -> Dict:
    """
    Statistical comparison of degree distributions between two graphs.
    """
    degrees1 = [d for n, d in G1.degree()]
    degrees2 = [d for n, d in G2.degree()]
    
    if len(degrees1) == 0 or len(degrees2) == 0:
        return {
            'ks_statistic': 0,
            'ks_pvalue': 1.0,
            'distributions_similar': True
        }
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(degrees1, degrees2)
    
    # Mann-Whitney U test (non-parametric)
    mw_stat, mw_pval = stats.mannwhitneyu(degrees1, degrees2, alternative='two-sided')
    
    results = {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mw_statistic': mw_stat,
        'mw_pvalue': mw_pval,
        'distributions_similar': ks_pval > 0.05,  # Not significantly different
        f'{label1}_mean_degree': np.mean(degrees1),
        f'{label2}_mean_degree': np.mean(degrees2),
        f'{label1}_median_degree': np.median(degrees1),
        f'{label2}_median_degree': np.median(degrees2),
    }
    
    return results


def comprehensive_graph_analysis(G: nx.Graph, graph_name: str = "Graph") -> pd.DataFrame:
    """
    Run all analysis functions and return comprehensive results.
    """
    print(f"\n  Analyzing {graph_name}...")
    
    all_metrics = {}
    
    # Basic metrics
    all_metrics['graph_name'] = graph_name
    all_metrics['num_nodes'] = G.number_of_nodes()
    all_metrics['num_edges'] = G.number_of_edges()
    all_metrics['density'] = nx.density(G)
    
    # Centrality
    print(f"    - Calculating centrality metrics...")
    all_metrics.update(calculate_centrality_metrics(G))
    
    # Community structure
    print(f"    - Detecting communities...")
    all_metrics.update(analyze_community_structure(G))
    
    # Assortativity
    print(f"    - Calculating assortativity...")
    all_metrics.update(calculate_assortativity(G))
    
    # Path lengths
    print(f"    - Analyzing path lengths...")
    all_metrics.update(analyze_path_lengths(G))
    
    # Robustness
    print(f"    - Measuring robustness...")
    all_metrics.update(calculate_robustness_metrics(G))
    
    # Small-world
    print(f"    - Checking small-world properties...")
    all_metrics.update(analyze_small_world_properties(G))
    
    # ADD THIS NEW LINE:
    # Graph coloring
    print(f"    - Analyzing graph coloring properties...")
    all_metrics.update(analyze_graph_coloring(G))
    
    return pd.DataFrame([all_metrics])


def run_statistical_tests(graphs_dict: Dict[str, Tuple[nx.Graph, dict]]) -> pd.DataFrame:
    """
    Run statistical tests comparing fake vs real news in each domain.
    """
    print("\n  Running statistical comparisons...")
    
    results = []
    
    # Compare GossipCop: Fake vs Real
    G_gf, _ = graphs_dict['gossip_fake']
    G_gr, _ = graphs_dict['gossip_real']
    
    gossip_comparison = compare_degree_distributions(
        G_gf, G_gr, "GossipCop_Fake", "GossipCop_Real"
    )
    gossip_comparison['domain'] = 'GossipCop'
    results.append(gossip_comparison)
    
    # Compare PolitiFact: Fake vs Real
    G_pf, _ = graphs_dict['politi_fake']
    G_pr, _ = graphs_dict['politi_real']
    
    politi_comparison = compare_degree_distributions(
        G_pf, G_pr, "PolitiFact_Fake", "PolitiFact_Real"
    )
    politi_comparison['domain'] = 'PolitiFact'
    results.append(politi_comparison)
    
    return pd.DataFrame(results)


def generate_insight_report(all_results: pd.DataFrame, 
                           statistical_tests: pd.DataFrame,
                           similarity_results: pd.DataFrame = None) -> str:
    """
    Generate a text report with key insights.
    """
    report = []
    report.append("\n" + "="*80)
    report.append(" ADVANCED NETWORK ANALYSIS INSIGHTS ".center(80, "="))
    report.append("="*80 + "\n")
    
    # ... existing code ...
    
    # ADD THIS NEW SECTION at the end before the final separator:
    
    # Graph Coloring Analysis
    report.append("\n6. GRAPH COLORING (Independent News Groups):")
    for _, row in all_results.iterrows():
        chromatic = row.get('chromatic_number_estimate', 0)
        max_indep = row.get('max_independent_set_size', 0)
        report.append(f"   {row['graph_name']:20s}: "
                     f"Colors needed={int(chromatic)}, "
                     f"Max independent set={int(max_indep)}")
    
    # Graph Similarity Analysis
    if similarity_results is not None and not similarity_results.empty:
        report.append("\n7. STRUCTURAL SIMILARITY (Graph Isomorphism):")
        for _, row in similarity_results.iterrows():
            domain = row['domain']
            comparison = row['comparison']
            sim_score = row.get('structural_similarity_score', 0)
            correlation = row.get('degree_sequence_correlation', 0)
            
            if sim_score < 0.3:
                similarity_level = "VERY SIMILAR"
            elif sim_score < 0.6:
                similarity_level = "MODERATELY SIMILAR"
            else:
                similarity_level = "VERY DIFFERENT"
            
            report.append(f"   {domain:15s} - {comparison:30s}: "
                         f"{similarity_level} (score={sim_score:.3f}, "
                         f"degree_corr={correlation:.3f})")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)
