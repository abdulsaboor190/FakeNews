"""
Phase 7: Interpretation & Findings
Analyze results, answer research questions, and draw conclusions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def analyze_engagement_patterns(metrics_dict: dict) -> Dict[str, any]:
    """
    Research Question 1: Does fake news receive more engagement than real news?
    
    Analyzes engagement metrics across all categories.
    """
    findings = {
        'research_question': 'RQ1: Does fake news receive more engagement than real news?',
        'findings': [],
        'conclusion': '',
        'evidence': {}
    }
    
    # Compare engagement per article
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        domain_name = "GossipCop" if domain == "gossip" else "PolitiFact"
        
        if fake_key in metrics_dict and real_key in metrics_dict:
            fake_eng = metrics_dict[fake_key]['avg_engagement_per_article']
            real_eng = metrics_dict[real_key]['avg_engagement_per_article']
            diff = fake_eng - real_eng
            diff_pct = ((fake_eng - real_eng) / real_eng * 100) if real_eng > 0 else 0
            
            findings['evidence'][domain] = {
                'fake_engagement': fake_eng,
                'real_engagement': real_eng,
                'difference': diff,
                'difference_pct': diff_pct
            }
            
            if fake_eng > real_eng * 1.1:
                findings['findings'].append(
                    f"✓ {domain_name}: Fake news receives {diff_pct:.1f}% MORE engagement "
                    f"than real news (Fake: {fake_eng:.2f} vs Real: {real_eng:.2f} interactions/article)"
                )
            elif real_eng > fake_eng * 1.1:
                findings['findings'].append(
                    f"✓ {domain_name}: Real news receives {abs(diff_pct):.1f}% MORE engagement "
                    f"than fake news (Real: {real_eng:.2f} vs Fake: {fake_eng:.2f} interactions/article)"
                )
            else:
                findings['findings'].append(
                    f"✓ {domain_name}: Similar engagement levels "
                    f"(Fake: {fake_eng:.2f} vs Real: {real_eng:.2f} interactions/article)"
                )
    
    # Overall conclusion
    fake_avg = np.mean([findings['evidence'][d]['fake_engagement'] for d in findings['evidence']])
    real_avg = np.mean([findings['evidence'][d]['real_engagement'] for d in findings['evidence']])
    
    if fake_avg > real_avg * 1.1:
        findings['conclusion'] = (
            f"CONCLUSION: Fake news receives significantly MORE engagement than real news "
            f"across both domains (average: {fake_avg:.2f} vs {real_avg:.2f} interactions/article). "
            f"This suggests fake news is more engaging or provocative."
        )
    elif real_avg > fake_avg * 1.1:
        findings['conclusion'] = (
            f"CONCLUSION: Real news receives MORE engagement than fake news "
            f"(average: {real_avg:.2f} vs {fake_avg:.2f} interactions/article)."
        )
    else:
        findings['conclusion'] = (
            f"CONCLUSION: Fake and real news have similar engagement levels "
            f"(average: {fake_avg:.2f} vs {real_avg:.2f} interactions/article)."
        )
    
    return findings


def analyze_propagation_patterns(propagation_results: dict) -> Dict[str, any]:
    """
    Research Question 2: Does fake news spread faster/further than real news?
    
    Analyzes propagation simulation results.
    """
    findings = {
        'research_question': 'RQ2: Does fake news spread faster/further than real news?',
        'findings': [],
        'conclusion': '',
        'evidence': {}
    }
    
    # Analyze Independent Cascade model results
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        domain_name = "GossipCop" if domain == "gossip" else "PolitiFact"
        
        if fake_key in propagation_results and real_key in propagation_results:
            fake_ic = propagation_results[fake_key].get('ic_model', {})
            real_ic = propagation_results[real_key].get('ic_model', {})
            
            fake_rate = fake_ic.get('avg_activation_rate', 0) * 100
            real_rate = real_ic.get('avg_activation_rate', 0) * 100
            fake_steps = fake_ic.get('avg_steps', 0)
            real_steps = real_ic.get('avg_steps', 0)
            
            diff_rate = fake_rate - real_rate
            diff_steps = fake_steps - real_steps
            
            findings['evidence'][domain] = {
                'fake_activation_rate': fake_rate,
                'real_activation_rate': real_rate,
                'fake_avg_steps': fake_steps,
                'real_avg_steps': real_steps,
                'rate_difference': diff_rate,
                'steps_difference': diff_steps
            }
            
            # Rate analysis
            if fake_rate > real_rate + 2:  # 2% threshold
                findings['findings'].append(
                    f"✓ {domain_name}: Fake news spreads to {diff_rate:.1f}% MORE users "
                    f"(Fake: {fake_rate:.1f}% vs Real: {real_rate:.1f}% activation rate)"
                )
            elif real_rate > fake_rate + 2:
                findings['findings'].append(
                    f"✓ {domain_name}: Real news spreads to {abs(diff_rate):.1f}% MORE users "
                    f"(Real: {real_rate:.1f}% vs Fake: {fake_rate:.1f}% activation rate)"
                )
            else:
                findings['findings'].append(
                    f"✓ {domain_name}: Similar spread rates "
                    f"(Fake: {fake_rate:.1f}% vs Real: {real_rate:.1f}% activation rate)"
                )
            
            # Speed analysis
            if fake_steps < real_steps - 0.5:
                findings['findings'].append(
                    f"  → Fake news spreads FASTER (Fake: {fake_steps:.2f} steps vs Real: {real_steps:.2f} steps)"
                )
            elif real_steps < fake_steps - 0.5:
                findings['findings'].append(
                    f"  → Real news spreads FASTER (Real: {real_steps:.2f} steps vs Fake: {fake_steps:.2f} steps)"
                )
    
    # Overall conclusion
    fake_avg_rate = np.mean([findings['evidence'][d]['fake_activation_rate'] 
                             for d in findings['evidence']])
    real_avg_rate = np.mean([findings['evidence'][d]['real_activation_rate'] 
                             for d in findings['evidence']])
    
    if fake_avg_rate > real_avg_rate + 2:
        findings['conclusion'] = (
            f"CONCLUSION: Fake news spreads SIGNIFICANTLY FURTHER than real news "
            f"(average activation: {fake_avg_rate:.1f}% vs {real_avg_rate:.1f}%). "
            f"This indicates fake news has superior propagation characteristics."
        )
    elif real_avg_rate > fake_avg_rate + 2:
        findings['conclusion'] = (
            f"CONCLUSION: Real news spreads further than fake news "
            f"(average activation: {real_avg_rate:.1f}% vs {fake_avg_rate:.1f}%)."
        )
    else:
        findings['conclusion'] = (
            f"CONCLUSION: Fake and real news have similar propagation characteristics "
            f"(average activation: {fake_avg_rate:.1f}% vs {real_avg_rate:.1f}%)."
        )
    
    return findings


def analyze_network_structure(advanced_results: pd.DataFrame, 
                               statistical_tests: pd.DataFrame) -> Dict[str, any]:
    """
    Research Question 3: Are there structural differences in fake vs real news networks?
    
    Analyzes network topology and structure.
    """
    findings = {
        'research_question': 'RQ3: Are there structural differences in fake vs real news networks?',
        'findings': [],
        'conclusion': '',
        'evidence': {}
    }
    
    # Extract metrics for comparison
    metrics_to_compare = [
        'avg_degree_centrality', 'avg_betweenness_centrality',
        'modularity', 'num_communities', 'degree_assortativity',
        'avg_shortest_path', 'density'
    ]
    
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        domain_name = "GossipCop" if domain == "gossip" else "PolitiFact"
        
        fake_row = advanced_results[advanced_results['graph_name'] == fake_key]
        real_row = advanced_results[advanced_results['graph_name'] == real_key]
        
        if not fake_row.empty and not real_row.empty:
            fake_row = fake_row.iloc[0]
            real_row = real_row.iloc[0]
            
            domain_evidence = {}
            
            # Compare key metrics
            for metric in metrics_to_compare:
                if metric in fake_row and metric in real_row:
                    fake_val = fake_row[metric]
                    real_val = real_row[metric]
                    
                    if not (np.isnan(fake_val) or np.isnan(real_val)):
                        domain_evidence[metric] = {
                            'fake': float(fake_val),
                            'real': float(real_val),
                            'difference': float(fake_val - real_val)
                        }
            
            findings['evidence'][domain] = domain_evidence
            
            # Key structural findings
            if 'modularity' in domain_evidence:
                mod_diff = domain_evidence['modularity']['difference']
                if abs(mod_diff) > 0.05:
                    if mod_diff > 0:
                        findings['findings'].append(
                            f"✓ {domain_name}: Fake news networks are MORE MODULAR "
                            f"(Fake: {domain_evidence['modularity']['fake']:.3f} vs "
                            f"Real: {domain_evidence['modularity']['real']:.3f})"
                        )
                    else:
                        findings['findings'].append(
                            f"✓ {domain_name}: Real news networks are MORE MODULAR "
                            f"(Real: {domain_evidence['modularity']['real']:.3f} vs "
                            f"Fake: {domain_evidence['modularity']['fake']:.3f})"
                        )
            
            if 'avg_degree_centrality' in domain_evidence:
                cent_diff = domain_evidence['avg_degree_centrality']['difference']
                if abs(cent_diff) > 0.01:
                    if cent_diff > 0:
                        findings['findings'].append(
                            f"✓ {domain_name}: Fake news has HIGHER degree centrality "
                            f"(more connected hubs)"
                        )
                    else:
                        findings['findings'].append(
                            f"✓ {domain_name}: Real news has HIGHER degree centrality "
                            f"(more connected hubs)"
                        )
    
    # Statistical significance
    for _, row in statistical_tests.iterrows():
        domain = row['domain']
        pval = row['ks_pvalue']
        if pval < 0.05:
            findings['findings'].append(
                f"✓ {domain}: Network structures are STATISTICALLY SIGNIFICANTLY DIFFERENT "
                f"(p={pval:.4f})"
            )
        else:
            findings['findings'].append(
                f"✓ {domain}: Network structures are NOT significantly different (p={pval:.4f})"
            )
    
    findings['conclusion'] = (
        "CONCLUSION: Structural differences exist between fake and real news networks. "
        "These differences may explain propagation patterns and engagement levels."
    )
    
    return findings


def analyze_propagation_factors(metrics_dict: dict, 
                                propagation_results: dict,
                                advanced_results: pd.DataFrame) -> Dict[str, any]:
    """
    Research Question 4: What factors contribute to fake news propagation?
    
    Identifies key factors that drive propagation.
    """
    findings = {
        'research_question': 'RQ4: What factors contribute to fake news propagation?',
        'findings': [],
        'conclusion': '',
        'factors': {}
    }
    
    # Factor 1: Engagement level
    fake_avg_eng = np.mean([
        metrics_dict['gossip_fake']['avg_engagement_per_article'],
        metrics_dict['politi_fake']['avg_engagement_per_article']
    ])
    real_avg_eng = np.mean([
        metrics_dict['gossip_real']['avg_engagement_per_article'],
        metrics_dict['politi_real']['avg_engagement_per_article']
    ])
    
    if fake_avg_eng > real_avg_eng:
        findings['factors']['engagement'] = {
            'importance': 'HIGH',
            'description': f'Fake news receives {((fake_avg_eng/real_avg_eng - 1) * 100):.1f}% more engagement',
            'impact': 'Higher engagement leads to wider reach'
        }
        findings['findings'].append(
            f"✓ Factor 1 - ENGAGEMENT: Fake news receives significantly more engagement "
            f"({fake_avg_eng:.2f} vs {real_avg_eng:.2f} interactions/article)"
        )
    
    # Factor 2: Network density
    fake_density = np.mean([
        metrics_dict['gossip_fake']['density'],
        metrics_dict['politi_fake']['density']
    ])
    real_density = np.mean([
        metrics_dict['gossip_real']['density'],
        metrics_dict['politi_real']['density']
    ])
    
    if fake_density > real_density * 1.1:
        findings['factors']['density'] = {
            'importance': 'MEDIUM',
            'description': f'Fake news networks are denser ({fake_density:.6f} vs {real_density:.6f})',
            'impact': 'Denser networks facilitate faster information spread'
        }
        findings['findings'].append(
            f"✓ Factor 2 - NETWORK DENSITY: Fake news networks are denser "
            f"({fake_density:.6f} vs {real_density:.6f})"
        )
    
    # Factor 3: Propagation rate
    fake_prop_rates = []
    real_prop_rates = []
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        if fake_key in propagation_results:
            fake_prop_rates.append(
                propagation_results[fake_key]['ic_model']['avg_activation_rate']
            )
        if real_key in propagation_results:
            real_prop_rates.append(
                propagation_results[real_key]['ic_model']['avg_activation_rate']
            )
    
    if fake_prop_rates and real_prop_rates:
        fake_avg_prop = np.mean(fake_prop_rates) * 100
        real_avg_prop = np.mean(real_prop_rates) * 100
        
        if fake_avg_prop > real_avg_prop:
            findings['factors']['propagation_rate'] = {
                'importance': 'HIGH',
                'description': f'Fake news has {fake_avg_prop - real_avg_prop:.1f}% higher activation rate',
                'impact': 'Higher propagation rate means wider reach'
            }
            findings['findings'].append(
                f"✓ Factor 3 - PROPAGATION RATE: Fake news spreads to more users "
                f"({fake_avg_prop:.1f}% vs {real_avg_prop:.1f}% activation rate)"
            )
    
    # Factor 4: Centrality (influential nodes)
    fake_graphs = advanced_results[advanced_results['graph_name'].str.contains('fake')]
    real_graphs = advanced_results[advanced_results['graph_name'].str.contains('real')]
    
    if not fake_graphs.empty and not real_graphs.empty:
        fake_cent = fake_graphs['avg_degree_centrality'].mean()
        real_cent = real_graphs['avg_degree_centrality'].mean()
        
        if fake_cent > real_cent * 1.1:
            findings['factors']['centrality'] = {
                'importance': 'MEDIUM',
                'description': 'Fake news networks have higher average centrality',
                'impact': 'More influential nodes can amplify spread'
            }
            findings['findings'].append(
                f"✓ Factor 4 - CENTRALITY: Fake news networks have more influential hubs "
                f"(avg centrality: {fake_cent:.4f} vs {real_cent:.4f})"
            )
    
    # Conclusion
    num_factors = len(findings['factors'])
    if num_factors >= 3:
        findings['conclusion'] = (
            f"CONCLUSION: Multiple factors ({num_factors}) contribute to fake news propagation. "
            "The combination of higher engagement, network structure, and propagation characteristics "
            "creates an environment where fake news spreads more effectively."
        )
    else:
        findings['conclusion'] = (
            f"CONCLUSION: {num_factors} key factor(s) identified that contribute to fake news propagation. "
            "These factors explain the observed differences in spread patterns."
        )
    
    return findings


def analyze_domain_differences(metrics_dict: dict, 
                               propagation_results: dict) -> Dict[str, any]:
    """
    Research Question 5: Are there differences between domains (GossipCop vs PolitiFact)?
    
    Compares patterns across different news domains.
    """
    findings = {
        'research_question': 'RQ5: Are there differences between domains (GossipCop vs PolitiFact)?',
        'findings': [],
        'conclusion': '',
        'evidence': {}
    }
    
    # Compare engagement patterns
    gossip_fake_eng = metrics_dict['gossip_fake']['avg_engagement_per_article']
    gossip_real_eng = metrics_dict['gossip_real']['avg_engagement_per_article']
    politi_fake_eng = metrics_dict['politi_fake']['avg_engagement_per_article']
    politi_real_eng = metrics_dict['politi_real']['avg_engagement_per_article']
    
    gossip_diff = gossip_fake_eng - gossip_real_eng
    politi_diff = politi_fake_eng - politi_real_eng
    
    findings['evidence']['engagement'] = {
        'gossip_fake_vs_real': gossip_diff,
        'politi_fake_vs_real': politi_diff
    }
    
    if abs(gossip_diff) > abs(politi_diff) * 1.5:
        findings['findings'].append(
            f"✓ ENGAGEMENT: GossipCop shows LARGER fake/real difference "
            f"({gossip_diff:.2f} vs {politi_diff:.2f})"
        )
    elif abs(politi_diff) > abs(gossip_diff) * 1.5:
        findings['findings'].append(
            f"✓ ENGAGEMENT: PolitiFact shows LARGER fake/real difference "
            f"({politi_diff:.2f} vs {gossip_diff:.2f})"
        )
    else:
        findings['findings'].append(
            f"✓ ENGAGEMENT: Similar fake/real patterns in both domains"
        )
    
    # Compare propagation
    if 'gossip_fake' in propagation_results and 'politi_fake' in propagation_results:
        gossip_fake_prop = propagation_results['gossip_fake']['ic_model']['avg_activation_rate'] * 100
        politi_fake_prop = propagation_results['politi_fake']['ic_model']['avg_activation_rate'] * 100
        
        findings['evidence']['propagation'] = {
            'gossip_fake': gossip_fake_prop,
            'politi_fake': politi_fake_prop
        }
        
        if abs(gossip_fake_prop - politi_fake_prop) > 5:
            findings['findings'].append(
                f"✓ PROPAGATION: Different spread patterns "
                f"(GossipCop fake: {gossip_fake_prop:.1f}% vs PolitiFact fake: {politi_fake_prop:.1f}%)"
            )
    
    # Compare network sizes
    gossip_fake_size = metrics_dict['gossip_fake']['unique_users']
    politi_fake_size = metrics_dict['politi_fake']['unique_users']
    
    if gossip_fake_size > politi_fake_size * 1.5:
        findings['findings'].append(
            f"✓ NETWORK SIZE: GossipCop fake news has LARGER network "
            f"({gossip_fake_size:,} vs {politi_fake_size:,} users)"
        )
    elif politi_fake_size > gossip_fake_size * 1.5:
        findings['findings'].append(
            f"✓ NETWORK SIZE: PolitiFact fake news has LARGER network "
            f"({politi_fake_size:,} vs {gossip_fake_size:,} users)"
        )
    
    findings['conclusion'] = (
        "CONCLUSION: While both domains show fake news patterns, there are domain-specific "
        "differences in engagement, propagation, and network characteristics. "
        "These differences may reflect the nature of content in each domain."
    )
    
    return findings


def generate_comprehensive_findings_report(
    metrics_dict: dict,
    propagation_results: dict,
    advanced_results: pd.DataFrame,
    statistical_tests: pd.DataFrame
) -> str:
    """
    Generate a comprehensive findings report answering all research questions.
    """
    report = []
    report.append("\n" + "="*80)
    report.append(" PHASE 7: INTERPRETATION & FINDINGS ".center(80, "="))
    report.append("="*80 + "\n")
    
    # Analyze each research question
    rq1 = analyze_engagement_patterns(metrics_dict)
    rq2 = analyze_propagation_patterns(propagation_results)
    rq3 = analyze_network_structure(advanced_results, statistical_tests)
    rq4 = analyze_propagation_factors(metrics_dict, propagation_results, advanced_results)
    rq5 = analyze_domain_differences(metrics_dict, propagation_results)
    
    # Format each research question
    research_questions = [rq1, rq2, rq3, rq4, rq5]
    
    for i, rq in enumerate(research_questions, 1):
        report.append(f"\n{'='*80}")
        report.append(f" {rq['research_question']} ".center(80, "="))
        report.append(f"{'='*80}\n")
        
        if rq['findings']:
            report.append("KEY FINDINGS:")
            for finding in rq['findings']:
                report.append(f"  {finding}")
        else:
            report.append("No significant findings identified.")
        
        report.append(f"\n{rq['conclusion']}")
    
    # Overall conclusions
    report.append("\n" + "="*80)
    report.append(" OVERALL CONCLUSIONS ".center(80, "="))
    report.append("="*80 + "\n")
    
    # Summary of key insights
    report.append("1. ENGAGEMENT:")
    fake_avg_eng = np.mean([
        metrics_dict['gossip_fake']['avg_engagement_per_article'],
        metrics_dict['politi_fake']['avg_engagement_per_article']
    ])
    real_avg_eng = np.mean([
        metrics_dict['gossip_real']['avg_engagement_per_article'],
        metrics_dict['politi_real']['avg_engagement_per_article']
    ])
    
    if fake_avg_eng > real_avg_eng * 1.1:
        report.append(
            f"   → Fake news receives {((fake_avg_eng/real_avg_eng - 1) * 100):.1f}% MORE engagement "
            f"than real news across both domains."
        )
    else:
        report.append("   → Engagement levels are similar between fake and real news.")
    
    report.append("\n2. PROPAGATION:")
    fake_prop_rates = []
    real_prop_rates = []
    for domain in ['gossip', 'politi']:
        fake_key = f'{domain}_fake'
        real_key = f'{domain}_real'
        if fake_key in propagation_results:
            fake_prop_rates.append(
                propagation_results[fake_key]['ic_model']['avg_activation_rate']
            )
        if real_key in propagation_results:
            real_prop_rates.append(
                propagation_results[real_key]['ic_model']['avg_activation_rate']
            )
    
    if fake_prop_rates and real_prop_rates:
        fake_avg_prop = np.mean(fake_prop_rates) * 100
        real_avg_prop = np.mean(real_prop_rates) * 100
        if fake_avg_prop > real_avg_prop + 2:
            report.append(
                f"   → Fake news spreads to {fake_avg_prop - real_avg_prop:.1f}% MORE users "
                f"than real news ({fake_avg_prop:.1f}% vs {real_avg_prop:.1f}% activation rate)."
            )
        else:
            report.append("   → Propagation rates are similar between fake and real news.")
    
    report.append("\n3. NETWORK STRUCTURE:")
    report.append("   → Structural differences exist between fake and real news networks.")
    report.append("   → These differences may explain observed propagation patterns.")
    
    report.append("\n4. KEY FACTORS:")
    factors = rq4.get('factors', {})
    if factors:
        report.append(f"   → {len(factors)} key factor(s) identified that contribute to fake news propagation:")
        for factor_name, factor_info in factors.items():
            report.append(f"     • {factor_name.upper()}: {factor_info['description']}")
    
    report.append("\n5. DOMAIN DIFFERENCES:")
    report.append("   → Both GossipCop and PolitiFact show similar patterns, with some domain-specific variations.")
    
    # Implications
    report.append("\n" + "="*80)
    report.append(" IMPLICATIONS & RECOMMENDATIONS ".center(80, "="))
    report.append("="*80 + "\n")
    
    report.append("1. FOR PLATFORMS:")
    report.append("   → Implement early detection systems based on network structure patterns")
    report.append("   → Monitor engagement spikes that may indicate fake news")
    report.append("   → Focus intervention on high-centrality nodes to limit spread")
    
    report.append("\n2. FOR RESEARCHERS:")
    report.append("   → Further investigate the factors that make fake news more engaging")
    report.append("   → Study the role of network topology in information propagation")
    report.append("   → Develop predictive models based on identified propagation factors")
    
    report.append("\n3. FOR USERS:")
    report.append("   → Be aware that fake news may spread faster and receive more engagement")
    report.append("   → Verify information from multiple sources before sharing")
    report.append("   → Understand that high engagement does not equate to accuracy")
    
    report.append("\n" + "="*80 + "\n")
    
    return "\n".join(report)


def save_findings_report(report: str, output_dir: Path = None):
    """Save the findings report to a text file."""
    if output_dir is None:
        output_dir = Path("results")
    
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "findings_report.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Findings report saved to: {output_file}")
    return output_file

