#!/usr/bin/env python3
"""
Analyze raw LLM dataset statistics
Provides comprehensive statistics for Stars, Forks, Commits, Releases, Contributors, LOC
"""

import json
import pandas as pd
import numpy as np
from collections import Counter

def analyze_dataset(json_file_path):
    """Analyze the raw LLM dataset and provide comprehensive statistics"""
    
    # Load dataset
    print("Loading dataset...")
    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    # Extract basic info
    metadata = dataset.get('metadata', {})
    repositories = dataset.get('repositories', [])
    
    print("🚀 LLM Repository Dataset Analysis")
    print("=" * 50)
    
    # Dataset Overview
    print(f"📊 Dataset Overview:")
    print(f"   Total repositories attempted: {metadata.get('total_repositories_attempted', 'N/A')}")
    print(f"   Successful extractions: {metadata.get('successful_extractions', 'N/A')}")
    print(f"   Success rate: {metadata.get('success_rate_percent', 'N/A')}%")
    print(f"   Total raw issues extracted: {metadata.get('total_raw_issues_extracted', 'N/A')}")
    print(f"   Available repositories for analysis: {len(repositories)}")
    
    if not repositories:
        print("❌ No repositories found in dataset")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(repositories)
    
    # Classification Distribution
    print(f"\n🏗️ Classification Distribution:")
    if 'classification_scope' in df.columns:
        scope_counts = df['classification_scope'].value_counts()
        for scope, count in scope_counts.items():
            print(f"   {scope}: {count} repositories ({count/len(df)*100:.1f}%)")
    
    if 'classification_function' in df.columns:
        function_counts = df['classification_function'].value_counts()
        print(f"\n⚙️ Function Distribution:")
        for func, count in function_counts.items():
            print(f"   {func}: {count} repositories ({count/len(df)*100:.1f}%)")
    
    # Main Statistics
    print(f"\n📈 Repository Metrics Statistics:")
    print("=" * 50)
    
    metrics = ['stars', 'forks', 'total_commits', 'total_releases', 'total_contributors', 'total_loc']
    
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"\n⭐ {metric.upper().replace('_', ' ')}:")
                print(f"   Count: {len(values)}")
                print(f"   Mean: {values.mean():.2f}")
                print(f"   Median: {values.median():.2f}")
                print(f"   Std Dev: {values.std():.2f}")
                print(f"   Min: {values.min():,}")
                print(f"   Max: {values.max():,}")
                print(f"   25th percentile: {values.quantile(0.25):.2f}")
                print(f"   75th percentile: {values.quantile(0.75):.2f}")
    
    # Additional Repository Metrics
    additional_metrics = ['repo_age_months', 'active_contributors_3m', 'file_count']
    
    print(f"\n📊 Additional Repository Metrics:")
    for metric in additional_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"\n   {metric.replace('_', ' ').title()}:")
                print(f"     Mean: {values.mean():.2f}")
                print(f"     Median: {values.median():.2f}")
                print(f"     Range: {values.min():.2f} - {values.max():.2f}")
    
    # Issue Statistics
    print(f"\n📋 Issue Statistics:")
    issue_metrics = ['total_issues_analyzed', 'closed_issues_count']
    
    for metric in issue_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"\n   {metric.replace('_', ' ').title()}:")
                print(f"     Total across all repos: {values.sum():,}")
                print(f"     Mean per repo: {values.mean():.1f}")
                print(f"     Median per repo: {values.median():.1f}")
                print(f"     Max in single repo: {values.max():,}")
    
    # Calculated Metrics
    print(f"\n🧮 Calculated Metrics:")
    calc_metrics = ['issue_closure_rate', 'commits_per_month_avg', 'maintenance_load', 
                   'contributor_vitality', 'avg_loc_per_file']
    
    for metric in calc_metrics:
        if metric in df.columns:
            values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                print(f"\n   {metric.replace('_', ' ').title()}:")
                print(f"     Mean: {values.mean():.3f}")
                print(f"     Median: {values.median():.3f}")
                print(f"     Range: {values.min():.3f} - {values.max():.3f}")
    
    # Language Analysis
    print(f"\n💻 Programming Language Analysis:")
    all_languages = []
    
    for repo in repositories:
        lang_breakdown = repo.get('language_breakdown', {})
        for lang, percentage in lang_breakdown.items():
            all_languages.append(lang)
    
    if all_languages:
        lang_counts = Counter(all_languages)
        print(f"   Most common languages:")
        for lang, count in lang_counts.most_common(10):
            print(f"     {lang}: {count} repositories ({count/len(df)*100:.1f}%)")
    
    # Top Repositories by Category
    print(f"\n🌟 Top Repositories by Stars:")
    top_repos = df.nlargest(10, 'stars')[['repo_name', 'stars', 'classification_function']]
    for _, repo in top_repos.iterrows():
        print(f"   {repo['repo_name']}: {repo['stars']:,} stars ({repo['classification_function']})")
    
    # Repository Size Distribution
    print(f"\n📦 Repository Size Distribution (by Stars):")
    if 'stars' in df.columns:
        stars = df['stars'].dropna()
        if len(stars) > 0:
            small = len(stars[stars < 100])
            medium = len(stars[(stars >= 100) & (stars < 1000)])
            large = len(stars[(stars >= 1000) & (stars < 10000)])
            huge = len(stars[stars >= 10000])
            
            print(f"   Small (< 100 stars): {small} repos ({small/len(stars)*100:.1f}%)")
            print(f"   Medium (100-999 stars): {medium} repos ({medium/len(stars)*100:.1f}%)")
            print(f"   Large (1K-9.9K stars): {large} repos ({large/len(stars)*100:.1f}%)")
            print(f"   Huge (10K+ stars): {huge} repos ({huge/len(stars)*100:.1f}%)")
    
    # Quality Assessment
    print(f"\n✅ Data Quality Assessment:")
    if 'data_quality_score' in df.columns:
        quality_scores = df['data_quality_score'].dropna()
        if len(quality_scores) > 0:
            print(f"   Average data quality score: {quality_scores.mean():.3f}")
            print(f"   Repositories with quality > 0.8: {len(quality_scores[quality_scores > 0.8])}")
            print(f"   Repositories with quality > 0.9: {len(quality_scores[quality_scores > 0.9])}")
    
    # Function vs Scope Analysis
    if 'classification_scope' in df.columns and 'classification_function' in df.columns:
        print(f"\n🏗️ Scope vs Function Cross-Analysis:")
        crosstab = pd.crosstab(df['classification_scope'], df['classification_function'])
        print(crosstab)
        
        # Average stars by category
        print(f"\n⭐ Average Stars by Category:")
        for scope in df['classification_scope'].unique():
            scope_df = df[df['classification_scope'] == scope]
            if len(scope_df) > 0 and 'stars' in df.columns:
                avg_stars = scope_df['stars'].mean()
                print(f"   {scope}: {avg_stars:.0f} average stars")
                
                for func in scope_df['classification_function'].unique():
                    func_df = scope_df[scope_df['classification_function'] == func]
                    if len(func_df) > 0:
                        func_avg_stars = func_df['stars'].mean()
                        print(f"     └─ {func}: {func_avg_stars:.0f} average stars ({len(func_df)} repos)")

# Main execution
if __name__ == "__main__":
    # Analyze the dataset
    json_file_path = "/home/niruthi/corrected_repo/raw_llm_dataset.json"
    
    try:
        analyze_dataset(json_file_path)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")