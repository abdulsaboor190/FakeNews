import networkx as nx
import numpy as np
from typing import List, Set, Tuple
import random


class IndependentCascadeModel:
    """
    Independent Cascade propagation model for information spread.
    
    Each activated node gets ONE chance to activate each inactive neighbor
    with probability p.
    """
    
    def __init__(self, G: nx.Graph, propagation_prob: float = 0.1):
        """
        Args:
            G: NetworkX graph (bipartite or projected)
            propagation_prob: Probability of propagation along an edge
        """
        self.G = G
        self.p = propagation_prob
    
    def simulate(self, seed_nodes: List[str], max_steps: int = 6) -> dict:
        """
        Simulate propagation starting from seed nodes.
        
        Returns:
            dict with propagation statistics
        """
        # Track states
        active = set(seed_nodes)
        newly_active = set(seed_nodes)
        
        # Track propagation over time
        timeline = [len(active)]
        
        for step in range(max_steps):
            if not newly_active:
                break
            
            next_active = set()
            
            for node in newly_active:
                # Try to activate neighbors
                for neighbor in self.G.neighbors(node):
                    if neighbor not in active:
                        # Propagation happens with probability p
                        if random.random() < self.p:
                            next_active.add(neighbor)
            
            # Update states
            active.update(next_active)
            newly_active = next_active
            timeline.append(len(active))
        
        return {
            "total_activated": len(active),
            "activation_rate": len(active) / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0,
            "steps_taken": len(timeline) - 1,
            "timeline": timeline,
            "final_active_nodes": active
        }


class LinearThresholdModel:
    """
    Linear Threshold model: a node activates if enough of its neighbors are active.
    
    Each node has a threshold; it activates when the sum of weights from 
    active neighbors exceeds this threshold.
    """
    
    def __init__(self, G: nx.Graph, threshold: float = 0.3):
        """
        Args:
            G: NetworkX graph
            threshold: Fraction of neighbors that must be active
        """
        self.G = G
        self.threshold = threshold
    
    def simulate(self, seed_nodes: List[str], max_steps: int = 6) -> dict:
        """Simulate Linear Threshold propagation."""
        active = set(seed_nodes)
        timeline = [len(active)]
        
        for step in range(max_steps):
            new_active = set()
            
            for node in self.G.nodes():
                if node not in active:
                    neighbors = list(self.G.neighbors(node))
                    if not neighbors:
                        continue
                    
                    # Count active neighbors
                    active_neighbors = sum(1 for n in neighbors if n in active)
                    
                    # Activate if threshold is met
                    if active_neighbors / len(neighbors) >= self.threshold:
                        new_active.add(node)
            
            if not new_active:
                break
            
            active.update(new_active)
            timeline.append(len(active))
        
        return {
            "total_activated": len(active),
            "activation_rate": len(active) / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0,
            "steps_taken": len(timeline) - 1,
            "timeline": timeline,
            "final_active_nodes": active
        }


def select_top_degree_seeds(G: nx.Graph, num_seeds: int, node_type: str = None) -> List[str]:
    """
    Select seed nodes based on highest degree.
    
    Args:
        G: Graph
        num_seeds: Number of seeds to select
        node_type: If bipartite, specify 'news' or 'user'
    """
    if node_type:
        nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == node_type]
    else:
        nodes = list(G.nodes())
    
    # Sort by degree
    nodes_with_degree = [(n, G.degree(n)) for n in nodes]
    nodes_with_degree.sort(key=lambda x: x[1], reverse=True)
    
    return [n for n, _ in nodes_with_degree[:num_seeds]]


def run_multiple_simulations(G: nx.Graph, 
                             model_class,
                             num_runs: int = 50,
                             num_seeds: int = 5,
                             **model_kwargs) -> dict:
    """
    Run multiple simulations and aggregate results.
    
    Returns:
        dict with aggregated statistics
    """
    results = []
    
    for run in range(num_runs):
        # Select random seeds
        if G.number_of_nodes() > 0:
            seed_nodes = random.sample(list(G.nodes()), 
                                      min(num_seeds, G.number_of_nodes()))
            
            model = model_class(G, **model_kwargs)
            result = model.simulate(seed_nodes)
            results.append(result)
    
    if not results:
        return {
            "avg_activation_rate": 0,
            "std_activation_rate": 0,
            "avg_steps": 0,
            "all_results": []
        }
    
    # Aggregate
    activation_rates = [r["activation_rate"] for r in results]
    steps = [r["steps_taken"] for r in results]
    
    return {
        "avg_activation_rate": np.mean(activation_rates),
        "std_activation_rate": np.std(activation_rates),
        "avg_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "all_results": results
    }




def compare_propagation(
    graphs_dict: dict,
    propagation_prob: float = 0.1,
    num_runs: int = 20,
    max_users_for_projection: int = 3000,
) -> dict:
    """
    Compare propagation across all four graph categories.
    
    Args:
        graphs_dict: Dict with keys like 'gossip_fake', values are (Graph, metrics) tuples
        propagation_prob: Propagation probability for IC model
        num_runs: Number of simulation runs per graph
        max_users_for_projection: Maximum number of users to include in projection
    
    Returns:
        dict with propagation results for each category
    """
    propagation_results = {}
    
    for category, (G, _) in graphs_dict.items():
        print(f"\nSimulating propagation for {category}...")
        
        # Create user projection for realistic propagation
        user_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"}
        if len(user_nodes) > 0:
            # If there are too many users, keep only the top-degree ones to avoid
            # an enormous and slow projection step.
            if len(user_nodes) > max_users_for_projection:
                print(
                    f"  [info] {category}: {len(user_nodes)} users → "
                    f"downsampling to top {max_users_for_projection} by degree for projection"
                )
                # Compute degrees on bipartite graph and keep top users
                user_degrees = [(u, G.degree(u)) for u in user_nodes]
                user_degrees.sort(key=lambda x: x[1], reverse=True)
                keep_users = {u for u, _ in user_degrees[:max_users_for_projection]}
            else:
                keep_users = user_nodes

            # Use memory-efficient projection instead of NetworkX's built-in
            from analysis import build_memory_efficient_user_projection
            print(f"  [info] Building user projection (memory-efficient method)...")
            try:
                user_graph = build_memory_efficient_user_projection(G, keep_users, max_edges=500000)
                print(f"  [info] Projection complete: {user_graph.number_of_nodes()} nodes, "
                      f"{user_graph.number_of_edges()} edges")
            except MemoryError:
                print(f"  [warning] Memory error with {len(keep_users)} users, reducing further...")
                # Further reduce if still too large
                keep_users = {u for u, _ in user_degrees[:max_users_for_projection // 2]}
                user_graph = build_memory_efficient_user_projection(G, keep_users, max_edges=200000)
                print(f"  [info] Reduced projection: {user_graph.number_of_nodes()} nodes, "
                      f"{user_graph.number_of_edges()} edges")
        else:
            user_graph = G
        
        # Skip if graph is empty or too small
        if user_graph.number_of_nodes() < 5:
            print(f"  Skipping {category}: insufficient nodes")
            propagation_results[category] = {
                "ic_model": {"avg_activation_rate": 0, "std_activation_rate": 0},
                "lt_model": {"avg_activation_rate": 0, "std_activation_rate": 0}
            }
            continue
        
        # Run Independent Cascade
        ic_results = run_multiple_simulations(
            user_graph,
            IndependentCascadeModel,
            num_runs=num_runs,
            num_seeds=min(5, user_graph.number_of_nodes()),
            propagation_prob=propagation_prob
        )
        
        # Run Linear Threshold
        lt_results = run_multiple_simulations(
            user_graph,
            LinearThresholdModel,
            num_runs=num_runs,
            num_seeds=min(5, user_graph.number_of_nodes()),
            threshold=0.3
        )
        
        propagation_results[category] = {
            "ic_model": ic_results,
            "lt_model": lt_results
        }
        
        print(f"  IC Model: {ic_results['avg_activation_rate']:.2%} ± {ic_results['std_activation_rate']:.2%}")
        print(f"  LT Model: {lt_results['avg_activation_rate']:.2%} ± {lt_results['std_activation_rate']:.2%}")
    
    return propagation_results


def analyze_influential_spreaders(G: nx.Graph, top_n: int = 10) -> dict:
    """
    Identify most influential nodes for spreading information.
    
    Uses multiple centrality measures.
    """
    if G.number_of_nodes() == 0:
        return {}
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # Betweenness centrality (expensive for large graphs)
    if G.number_of_nodes() < 1000:
        between_cent = nx.betweenness_centrality(G)
    else:
        between_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    
    # Closeness centrality
    if nx.is_connected(G):
        close_cent = nx.closeness_centrality(G)
    else:
        close_cent = {}
    
    # Get top nodes by degree
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return {
        "top_degree": top_degree,
        "top_betweenness": top_between,
        "avg_degree_centrality": np.mean(list(degree_cent.values())),
        "avg_betweenness_centrality": np.mean(list(between_cent.values()))
    }