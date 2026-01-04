import pandas as pd
import networkx as nx
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Set, Dict, Tuple
import random


DATASET_DIR = Path("dataset")


def load_dataset(name: str) -> pd.DataFrame:
    """Load one of the four CSVs as a DataFrame."""
    path = DATASET_DIR / name
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["id"]).dropna(subset=["id", "tweet_ids"])
    return df


def explode_tweet_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Convert space/tab-separated tweet_ids into one row per (news_id, tweet_id)."""
    interactions = (
        df.assign(tweet_ids=df["tweet_ids"].astype(str))
        .loc[df["tweet_ids"].notna() & (df["tweet_ids"].astype(str).str.len() > 0)]
    )
    interactions = interactions.assign(
        tweet_id_list=interactions["tweet_ids"].str.split(r"\s+")
    )
    interactions = interactions.explode("tweet_id_list")
    interactions = interactions.rename(
        columns={"id": "news_id", "tweet_id_list": "tweet_id"}
    )
    interactions = interactions[["news_id", "tweet_id"]]
    interactions = interactions.dropna(subset=["tweet_id"])
    interactions = interactions[interactions["tweet_id"].astype(str).str.len() > 0]
    return interactions


def build_bipartite_graph(interactions: pd.DataFrame) -> nx.Graph:
    """Build a bipartite graph: news articles ↔ users/tweets."""
    G = nx.Graph()

    for _, row in interactions.iterrows():
        news_node = f"news_{row['news_id']}"
        user_node = f"user_{row['tweet_id']}"

        if not G.has_node(news_node):
            G.add_node(news_node, bipartite="news")
        if not G.has_node(user_node):
            G.add_node(user_node, bipartite="user")

        if not G.has_edge(news_node, user_node):
            G.add_edge(news_node, user_node, weight=1)
        else:
            G[news_node][user_node]["weight"] += 1

    return G


def calculate_bipartite_metrics(G: nx.Graph) -> dict:
    """Calculate metrics for bipartite graph analysis."""
    news_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "news"}
    user_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"}
    
    metrics = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "news_articles": len(news_nodes),
        "unique_users": len(user_nodes),
        "avg_engagement_per_article": G.number_of_edges() / len(news_nodes) if news_nodes else 0,
        "density": nx.density(G),
    }
    
    # Degree statistics for news articles
    news_degrees = [G.degree(n) for n in news_nodes]
    if news_degrees:
        metrics["news_avg_degree"] = np.mean(news_degrees)
        metrics["news_max_degree"] = np.max(news_degrees)
        metrics["news_min_degree"] = np.min(news_degrees)
        metrics["news_std_degree"] = np.std(news_degrees)
    
    # Degree statistics for users
    user_degrees = [G.degree(n) for n in user_nodes]
    if user_degrees:
        metrics["user_avg_degree"] = np.mean(user_degrees)
        metrics["user_max_degree"] = np.max(user_degrees)
    
    # Connected components
    metrics["num_components"] = nx.number_connected_components(G)
    
    # Clustering (for bipartite, we project to one mode)
    if len(news_nodes) > 0:
        try:
            news_projection = nx.bipartite.projected_graph(G, news_nodes)
            metrics["news_clustering"] = nx.average_clustering(news_projection)
        except:
            metrics["news_clustering"] = 0
    
    return metrics


def build_memory_efficient_user_projection(G: nx.Graph, user_nodes: Set[str], 
                                           max_edges: int = 500000) -> nx.Graph:
    """
    Build a weighted user projection more memory-efficiently.
    
    This function builds the projection incrementally to avoid memory issues.
    It processes news articles one at a time without creating large intermediate structures.
    
    Args:
        G: Bipartite graph
        user_nodes: Set of user nodes to project
        max_edges: Maximum number of edges to create (to prevent memory issues)
    
    Returns:
        Projected user-user graph
    """
    # Build projection incrementally using a dictionary to count shared articles
    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    
    # Get all news nodes that are connected to our user nodes
    news_nodes = set()
    for u in user_nodes:
        if u in G:
            for neighbor in G.neighbors(u):
                if G.nodes[neighbor].get("bipartite") == "news":
                    news_nodes.add(neighbor)
    
    edges_created = 0
    max_users_per_article = 500  # Limit users per article to prevent quadratic explosion
    
    for news in news_nodes:
        if edges_created >= max_edges:
            break
            
        # Get all users connected to this news article (only from our user set)
        connected_users = [u for u in G.neighbors(news) if u in user_nodes]
        
        # Limit the number of users per article to prevent memory explosion
        if len(connected_users) > max_users_per_article:
            # Sample users to keep projection manageable
            connected_users = random.sample(connected_users, max_users_per_article)
        
        # Create edges between all pairs of users who shared this article
        for i, u1 in enumerate(connected_users):
            if edges_created >= max_edges:
                break
            for u2 in connected_users[i+1:]:
                if edges_created >= max_edges:
                    break
                # Use sorted tuple for undirected graph
                edge_key = tuple(sorted([u1, u2]))
                edge_weights[edge_key] += 1
                edges_created += 1
    
    # Build the graph from edge weights
    user_graph = nx.Graph()
    user_graph.add_nodes_from(user_nodes)
    
    # Add edges in batches to be more memory-efficient
    batch_size = 10000
    edge_items = list(edge_weights.items())
    for i in range(0, len(edge_items), batch_size):
        batch = edge_items[i:i+batch_size]
        for (u1, u2), weight in batch:
            user_graph.add_edge(u1, u2, weight=weight)
    
    return user_graph


def build_user_projection(G: nx.Graph, max_users: int = 5000, max_edges: int = 500000) -> nx.Graph:
    """
    Project bipartite graph to user-user network.
    Two users are connected if they engaged with the same news article.
    Edge weight = number of shared articles.
    
    Uses memory-efficient projection for large graphs.
    """
    user_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"}
    
    # If too many users, sample top-degree ones
    if len(user_nodes) > max_users:
        user_degrees = [(u, G.degree(u)) for u in user_nodes]
        user_degrees.sort(key=lambda x: x[1], reverse=True)
        user_nodes = {u for u, _ in user_degrees[:max_users]}
    
    # Use memory-efficient projection
    try:
        return build_memory_efficient_user_projection(G, user_nodes, max_edges=max_edges)
    except MemoryError:
        # If still fails, reduce further
        user_degrees = [(u, G.degree(u)) for u in user_nodes]
        user_degrees.sort(key=lambda x: x[1], reverse=True)
        user_nodes = {u for u, _ in user_degrees[:max_users // 2]}
        return build_memory_efficient_user_projection(G, user_nodes, max_edges=max_edges // 2)


def build_news_projection(G: nx.Graph) -> nx.Graph:
    """
    Project bipartite graph to news-news network.
    Two news articles are connected if shared by same users.
    Edge weight = number of shared users.
    """
    news_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "news"}
    return nx.bipartite.weighted_projected_graph(G, news_nodes)


def calculate_propagation_metrics(G: nx.Graph) -> dict:
    """
    Calculate metrics that simulate propagation characteristics.
    """
    news_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "news"}
    
    # Create news projection to analyze spread patterns
    news_proj = build_news_projection(G)
    
    metrics = {
        "news_network_density": nx.density(news_proj) if news_proj.number_of_nodes() > 1 else 0,
    }
    
    # Average shortest path (connectivity measure)
    if nx.is_connected(news_proj) and news_proj.number_of_nodes() > 1:
        metrics["news_avg_shortest_path"] = nx.average_shortest_path_length(news_proj)
    else:
        # For disconnected graphs, calculate for largest component
        if news_proj.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(news_proj), key=len)
            subgraph = news_proj.subgraph(largest_cc)
            if subgraph.number_of_nodes() > 1:
                metrics["news_avg_shortest_path"] = nx.average_shortest_path_length(subgraph)
            else:
                metrics["news_avg_shortest_path"] = 0
        else:
            metrics["news_avg_shortest_path"] = 0
    
    # Degree centrality (which news items are most central?)
    if news_proj.number_of_nodes() > 0:
        centrality = nx.degree_centrality(news_proj)
        metrics["news_avg_centrality"] = np.mean(list(centrality.values()))
    
    return metrics


def compare_fake_vs_real(fake_metrics: dict, real_metrics: dict, domain: str) -> None:
    """Print comparison table for fake vs real news in a domain."""
    print(f"\n{'='*70}")
    print(f"  {domain.upper()} DOMAIN: FAKE vs REAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Fake':<15} {'Real':<15} {'Difference'}")
    print(f"{'-'*70}")
    
    for key in fake_metrics:
        if key in real_metrics:
            fake_val = fake_metrics[key]
            real_val = real_metrics[key]
            diff = fake_val - real_val
            diff_pct = (diff / real_val * 100) if real_val != 0 else 0
            
            print(f"{key:<35} {fake_val:<15.4f} {real_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)")


def main() -> None:
    print("Loading datasets...")
    gossip_fake = load_dataset("gossipcop_fake.csv")
    gossip_real = load_dataset("gossipcop_real.csv")
    politi_fake = load_dataset("politifact_fake.csv")
    politi_real = load_dataset("politifact_real.csv")

    print("Creating interaction tables...")
    gossip_fake_edges = explode_tweet_ids(gossip_fake)
    gossip_real_edges = explode_tweet_ids(gossip_real)
    politi_fake_edges = explode_tweet_ids(politi_fake)
    politi_real_edges = explode_tweet_ids(politi_real)

    print("Building bipartite graphs...")
    G_gossip_fake = build_bipartite_graph(gossip_fake_edges)
    G_gossip_real = build_bipartite_graph(gossip_real_edges)
    G_politi_fake = build_bipartite_graph(politi_fake_edges)
    G_politi_real = build_bipartite_graph(politi_real_edges)

    print("\nCalculating comprehensive metrics...")
    
    # Calculate metrics for all graphs
    gf_metrics = calculate_bipartite_metrics(G_gossip_fake)
    gf_metrics.update(calculate_propagation_metrics(G_gossip_fake))
    
    gr_metrics = calculate_bipartite_metrics(G_gossip_real)
    gr_metrics.update(calculate_propagation_metrics(G_gossip_real))
    
    pf_metrics = calculate_bipartite_metrics(G_politi_fake)
    pf_metrics.update(calculate_propagation_metrics(G_politi_fake))
    
    pr_metrics = calculate_bipartite_metrics(G_politi_real)
    pr_metrics.update(calculate_propagation_metrics(G_politi_real))
    
    # Compare fake vs real within each domain
    compare_fake_vs_real(gf_metrics, gr_metrics, "GossipCop")
    compare_fake_vs_real(pf_metrics, pr_metrics, "PolitiFact")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"GossipCop Fake:  {gf_metrics['news_articles']} articles, {gf_metrics['unique_users']} users")
    print(f"GossipCop Real:  {gr_metrics['news_articles']} articles, {gr_metrics['unique_users']} users")
    print(f"PolitiFact Fake: {pf_metrics['news_articles']} articles, {pf_metrics['unique_users']} users")
    print(f"PolitiFact Real: {pr_metrics['news_articles']} articles, {pr_metrics['unique_users']} users")
    
    return {
        "gossip_fake": (G_gossip_fake, gf_metrics),
        "gossip_real": (G_gossip_real, gr_metrics),
        "politi_fake": (G_politi_fake, pf_metrics),
        "politi_real": (G_politi_real, pr_metrics),
    }


if __name__ == "__main__":
    results = main()