"""
Fake News Propagation Analysis - Complete Pipeline
Phases 1-7: From data loading to comprehensive analysis, visualization, and interpretation

Run this script to execute the complete analysis.
"""

import sys
from pathlib import Path
import pandas as pd

# Import Phase 1-3 modules (basic setup)
from analysis import (
    load_dataset, explode_tweet_ids, build_bipartite_graph,
    calculate_bipartite_metrics, calculate_propagation_metrics,
    build_user_projection, build_news_projection
)

# Import Phase 4 module (propagation simulation)
from propagation import compare_propagation, analyze_influential_spreaders

# Import Phase 5 module (advanced metrics)
from advanced_metrics import (
    comprehensive_graph_analysis, run_statistical_tests,
    generate_insight_report, run_graph_similarity_analysis
)

# Import Phase 6 modules (visualization)
from visualizer import (
    plot_degree_distribution, plot_network_sample,
    plot_metrics_comparison, plot_engagement_distribution,
    create_summary_table
)

from propagation_viz import (
    plot_propagation_timeline, plot_activation_rates,
    plot_propagation_speed, plot_propagation_heatmap,
    plot_fake_vs_real_comparison
)

# Import Phase 7 module (interpretation & findings)
from interpretation import (
    generate_comprehensive_findings_report,
    save_findings_report
)


def print_phase_header(phase_num: int, phase_name: str):
    """Print formatted phase header."""
    print("\n" + "="*80)
    print(f" PHASE {phase_num}: {phase_name} ".center(80, "="))
    print("="*80 + "\n")


def save_all_results(graphs_dict: dict, metrics_dict: dict, 
                    propagation_results: dict, advanced_results: pd.DataFrame,
                    statistical_tests: pd.DataFrame, similarity_results: pd.DataFrame = None):
    """Save all results to CSV files."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Basic metrics
    basic_metrics = []
    for category, metrics in metrics_dict.items():
        row = {"category": category}
        row.update(metrics)
        basic_metrics.append(row)
    pd.DataFrame(basic_metrics).to_csv(results_dir / "basic_metrics.csv", index=False)
    
    if similarity_results is not None:
        similarity_results.to_csv(results_dir / "graph_similarity.csv", index=False)
        print("  - graph_similarity.csv")
    # Advanced metrics
    advanced_results.to_csv(results_dir / "advanced_metrics.csv", index=False)
    
    # Statistical tests
    statistical_tests.to_csv(results_dir / "statistical_tests.csv", index=False)
    
    
    # Propagation results
    prop_data = []
    for category, results in propagation_results.items():
        for model in ['ic_model', 'lt_model']:
            if model in results:
                row = {
                    'category': category,
                    'model': model,
                    'avg_activation_rate': results[model]['avg_activation_rate'],
                    'std_activation_rate': results[model]['std_activation_rate'],
                    'avg_steps': results[model].get('avg_steps', 0),
                    'std_steps': results[model].get('std_steps', 0)
                }
                prop_data.append(row)
    pd.DataFrame(prop_data).to_csv(results_dir / "propagation_results.csv", index=False)
    
    print("\n✓ All results saved to results/ directory")
    print("  - basic_metrics.csv")
    print("  - advanced_metrics.csv")
    print("  - statistical_tests.csv")
    print("  - propagation_results.csv")


def print_executive_summary(metrics_dict: dict, propagation_results: dict, 
                           statistical_tests: pd.DataFrame):
    """Print executive summary of key findings."""
    print("\n" + "="*80)
    print(" EXECUTIVE SUMMARY ".center(80, "="))
    print("="*80 + "\n")
    
    print("1. DATASET OVERVIEW:")
    total_news = sum(m['news_articles'] for m in metrics_dict.values())
    total_users = sum(m['unique_users'] for m in metrics_dict.values())
    total_interactions = sum(m['total_edges'] for m in metrics_dict.values())
    print(f"   Total news articles: {total_news:,}")
    print(f"   Total unique users: {total_users:,}")
    print(f"   Total interactions: {total_interactions:,}")
    
    print("\n2. ENGAGEMENT COMPARISON (Average per Article):")
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        fake_eng = metrics_dict[fake_key]['avg_engagement_per_article']
        real_eng = metrics_dict[real_key]['avg_engagement_per_article']
        diff_pct = ((fake_eng - real_eng) / real_eng * 100) if real_eng > 0 else 0
        domain_name = "GossipCop" if domain == "gossip" else "PolitiFact"
        winner = "FAKE" if fake_eng > real_eng else "REAL"
        print(f"   {domain_name:12s}: Fake={fake_eng:.2f}, Real={real_eng:.2f} "
              f"({diff_pct:+.1f}%) → {winner} has more engagement")
    
    print("\n3. PROPAGATION SIMULATION (Independent Cascade):")
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        if fake_key in propagation_results and real_key in propagation_results:
            fake_rate = propagation_results[fake_key]['ic_model']['avg_activation_rate'] * 100
            real_rate = propagation_results[real_key]['ic_model']['avg_activation_rate'] * 100
            diff = fake_rate - real_rate
            domain_name = "GossipCop" if domain == "gossip" else "PolitiFact"
            winner = "FAKE" if fake_rate > real_rate else "REAL"
            print(f"   {domain_name:12s}: Fake={fake_rate:.1f}%, Real={real_rate:.1f}% "
                  f"({diff:+.1f}%) → {winner} spreads further")
    
    print("\n4. STATISTICAL SIGNIFICANCE:")
    for _, row in statistical_tests.iterrows():
        domain = row['domain']
        pval = row['ks_pvalue']
        if pval < 0.001:
            sig_level = "HIGHLY SIGNIFICANT (p<0.001)"
        elif pval < 0.05:
            sig_level = "SIGNIFICANT (p<0.05)"
        else:
            sig_level = "NOT SIGNIFICANT (p≥0.05)"
        print(f"   {domain:15s}: {sig_level}")
    
    print("\n5. KEY INSIGHT:")
    # Calculate overall fake vs real comparison
    fake_avg_eng = (metrics_dict['gossip_fake']['avg_engagement_per_article'] + 
                    metrics_dict['politi_fake']['avg_engagement_per_article']) / 2
    real_avg_eng = (metrics_dict['gossip_real']['avg_engagement_per_article'] + 
                    metrics_dict['politi_real']['avg_engagement_per_article']) / 2
    
    if fake_avg_eng > real_avg_eng * 1.1:
        print("   → Fake news tends to receive MORE engagement than real news")
    elif real_avg_eng > fake_avg_eng * 1.1:
        print("   → Real news tends to receive MORE engagement than fake news")
    else:
        print("   → Fake and real news have SIMILAR engagement levels")
    
    print("\n" + "="*80 + "\n")


def main():
    """Execute complete analysis pipeline."""
    
    print("\n" + "="*80)
    print(" FAKE NEWS PROPAGATION ANALYSIS ".center(80, "="))
    print(" Complete Pipeline: Phases 1-7 ".center(80))
    print("="*80 + "\n")
    
    # ============================================================================
    # PHASE 1-3: Data Loading and Graph Construction
    # ============================================================================
    print_phase_header(1, "DATA LOADING & GRAPH CONSTRUCTION")
    
    try:
        print("Loading datasets...")
        gossip_fake = load_dataset("gossipcop_fake.csv")
        gossip_real = load_dataset("gossipcop_real.csv")
        politi_fake = load_dataset("politifact_fake.csv")
        politi_real = load_dataset("politifact_real.csv")
        print("✓ Datasets loaded successfully")
        
        print("\nCreating interaction tables...")
        gossip_fake_edges = explode_tweet_ids(gossip_fake)
        gossip_real_edges = explode_tweet_ids(gossip_real)
        politi_fake_edges = explode_tweet_ids(politi_fake)
        politi_real_edges = explode_tweet_ids(politi_real)
        print("✓ Interaction tables created")
        
        print("\nBuilding bipartite graphs...")
        G_gossip_fake = build_bipartite_graph(gossip_fake_edges)
        G_gossip_real = build_bipartite_graph(gossip_real_edges)
        G_politi_fake = build_bipartite_graph(politi_fake_edges)
        G_politi_real = build_bipartite_graph(politi_real_edges)
        print("✓ All graphs constructed")
        
        print("\nCalculating basic metrics...")
        gf_metrics = calculate_bipartite_metrics(G_gossip_fake)
        gf_metrics.update(calculate_propagation_metrics(G_gossip_fake))
        
        gr_metrics = calculate_bipartite_metrics(G_gossip_real)
        gr_metrics.update(calculate_propagation_metrics(G_gossip_real))
        
        pf_metrics = calculate_bipartite_metrics(G_politi_fake)
        pf_metrics.update(calculate_propagation_metrics(G_politi_fake))
        
        pr_metrics = calculate_bipartite_metrics(G_politi_real)
        pr_metrics.update(calculate_propagation_metrics(G_politi_real))
        
        all_graphs = {
            'gossip_fake': (G_gossip_fake, gf_metrics),
            'gossip_real': (G_gossip_real, gr_metrics),
            'politi_fake': (G_politi_fake, pf_metrics),
            'politi_real': (G_politi_real, pr_metrics),
        }
        
        all_metrics = {
            'gossip_fake': gf_metrics,
            'gossip_real': gr_metrics,
            'politi_fake': pf_metrics,
            'politi_real': pr_metrics,
        }
        print("✓ Basic metrics calculated")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset files not found!")
        print("Please ensure all CSV files are in the 'dataset/' directory:")
        print("  - dataset/gossipcop_fake.csv")
        print("  - dataset/gossipcop_real.csv")
        print("  - dataset/politifact_fake.csv")
        print("  - dataset/politifact_real.csv")
        sys.exit(1)
    
    # ============================================================================
    # PHASE 4: Propagation Simulation
    # ============================================================================
    print_phase_header(4, "PROPAGATION SIMULATION")
    
    print("Running propagation simulations...")
    print("(This may take a few minutes, but should now finish in reasonable time.)\n")
    propagation_results = compare_propagation(
        all_graphs,
        propagation_prob=0.1,
        num_runs=20,
        max_users_for_projection=3000,
    )
    print("\n✓ Propagation simulations complete")
    
    # ============================================================================
    # PHASE 5: Advanced Network Analysis
    # ============================================================================
    print_phase_header(5, "ADVANCED NETWORK ANALYSIS")
    
    print("Performing comprehensive network analysis...")
    
    # Build user projections for analysis
    print("\nCreating user projection networks...")
    user_graphs = {}
    for key, (G, _) in all_graphs.items():
        user_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"}
        if len(user_nodes) > 0:
            user_graphs[key] = build_user_projection(G)
        else:
            user_graphs[key] = G
    
    # Run comprehensive analysis
    all_advanced_results = []
    for key, user_graph in user_graphs.items():
        result = comprehensive_graph_analysis(user_graph, key)
        all_advanced_results.append(result)
    
    advanced_results_df = pd.concat(all_advanced_results, ignore_index=True)
    
    print("\n✓ Advanced analysis complete")
    
# Statistical tests
    print("\nRunning statistical tests...")
    statistical_tests = run_statistical_tests(all_graphs)
    print("✓ Statistical tests complete")

    # Graph similarity analysis
    print("\nRunning graph similarity analysis...")
    similarity_results = run_graph_similarity_analysis(all_graphs)
    print("✓ Graph similarity analysis complete")

# Generate insights (UPDATE THIS LINE to include similarity_results)
    insight_report = generate_insight_report(advanced_results_df, statistical_tests, similarity_results)
    print(insight_report)
    # ============================================================================
    # PHASE 6: Visualization
    # ============================================================================
    print_phase_header(6, "VISUALIZATION GENERATION")
    
    print("Creating visualizations...")
    print("(This will generate multiple plots...)\n")
    
    # Basic network visualizations
    print("1. Degree distributions...")
    plot_degree_distribution(G_gossip_fake, "GossipCop Fake", "degree_gossip_fake.png")
    plot_degree_distribution(G_gossip_real, "GossipCop Real", "degree_gossip_real.png")
    plot_degree_distribution(G_politi_fake, "PolitiFact Fake", "degree_politi_fake.png")
    plot_degree_distribution(G_politi_real, "PolitiFact Real", "degree_politi_real.png")
    
    print("\n2. Network samples...")
    plot_network_sample(G_gossip_fake, "GossipCop Fake", "network_gossip_fake.png", max_nodes=150)
    plot_network_sample(G_gossip_real, "GossipCop Real", "network_gossip_real.png", max_nodes=150)
    plot_network_sample(G_politi_fake, "PolitiFact Fake", "network_politi_fake.png", max_nodes=150)
    plot_network_sample(G_politi_real, "PolitiFact Real", "network_politi_real.png", max_nodes=150)
    
    print("\n3. Comparison charts...")
    plot_metrics_comparison(all_metrics)
    plot_engagement_distribution(all_graphs)
    create_summary_table(all_metrics)
    
    print("\n4. Propagation visualizations...")
    plot_propagation_timeline(propagation_results)
    plot_activation_rates(propagation_results)
    plot_propagation_speed(propagation_results)
    plot_propagation_heatmap(propagation_results)
    plot_fake_vs_real_comparison(propagation_results)
    
    print("\n✓ All visualizations saved to results/figures/")
    
    # ============================================================================
    # PHASE 7: Interpretation & Findings
    # ============================================================================
    print_phase_header(7, "INTERPRETATION & FINDINGS")
    
    print("Analyzing results and generating comprehensive findings...")
    print("(Answering research questions and drawing conclusions...)\n")
    
    findings_report = generate_comprehensive_findings_report(
        all_metrics,
        propagation_results,
        advanced_results_df,
        statistical_tests
    )
    
    # Print the report
    print(findings_report)
    
    # Save the report
    save_findings_report(findings_report)
    
    print("\n✓ Phase 7 complete: Findings and interpretations generated")
    
    # ============================================================================
    # PHASE 8: Save All Results
    # ============================================================================
    print_phase_header(8, "SAVING RESULTS")
    save_all_results(all_graphs, all_metrics, propagation_results, 
                    advanced_results_df, statistical_tests, similarity_results)
    
    # ============================================================================
    # Executive Summary
    # ============================================================================
    print_executive_summary(all_metrics, propagation_results, statistical_tests)
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    print("="*80)
    print(" ANALYSIS COMPLETE! ".center(80, "="))
    print("="*80)
    print("\n📊 Results saved in:")
    print("   results/")
    print("   ├── basic_metrics.csv")
    print("   ├── advanced_metrics.csv")
    print("   ├── statistical_tests.csv")
    print("   ├── propagation_results.csv")
    print("   ├── findings_report.txt")
    print("   └── figures/")
    print("       ├── degree_*.png (8 files)")
    print("       ├── network_*.png (4 files)")
    print("       ├── metrics_comparison.png")
    print("       ├── engagement_distribution.png")
    print("       ├── summary_table.png")
    print("       ├── propagation_timeline.png")
    print("       ├── activation_rates.png")
    print("       ├── propagation_speed.png")
    print("       ├── propagation_heatmap.png")
    print("       └── fake_vs_real_propagation.png")
    print("\n🎉 Total: 4 CSV files + 1 findings report + 18 visualization files!")
    print("\n" + "="*80 + "\n")
    
    return all_graphs, all_metrics, propagation_results, advanced_results_df


if __name__ == "__main__":
    try:
        graphs, metrics, propagation, advanced = main()
        print("✅ SUCCESS: All phases completed!")
        print("\nYou can now:")
        print("  1. Review the findings report: results/findings_report.txt")
        print("  2. Review the CSV files for detailed metrics")
        print("  3. Examine the visualizations in results/figures/")
        print("  4. Use the data and findings to write your project report")
        print("  5. Import results in a notebook for further analysis")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)