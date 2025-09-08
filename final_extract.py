#!/usr/bin/env python3
"""
Raw LLM Repository Data Extraction Script
Extracts raw GitHub data only - no derived analysis or classifications
User will perform their own technical debt analysis later
"""

import requests
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

@dataclass
class RawIssue:
    """Raw issue data - no derived analysis"""
    number: int
    title: str
    body: str
    state: str
    created_at: str
    closed_at: Optional[str]
    updated_at: str
    labels: List[str]
    assignees: List[str]
    milestone: Optional[str]
    comments_count: int
    reactions: Dict[str, int]
    author: str
    html_url: str

@dataclass
class RawRepositoryData:
    """Raw repository data - minimal derived calculations"""
    
    # Identification
    repo_name: str
    url: str
    classification_scope: str
    classification_function: str
    
    # Raw GitHub metrics
    stars: int
    forks: int
    open_issues: int
    repo_age_months: float  # Only basic calculation needed
    total_commits: int
    total_releases: int
    total_contributors: int
    active_contributors_3m: int
    total_loc: int
    language_breakdown: Dict[str, float]
    file_count: int
    
    # Raw issue counts (basic aggregation only)
    total_issues_analyzed: int
    closed_issues_count: int
    
    # Basic calculated metrics (your finalized schema)
    issue_closure_rate: float
    commits_per_month_avg: float
    maintenance_load: float
    contributor_vitality: float
    avg_loc_per_file: float
    
    # Raw individual issues
    raw_issues: List[Dict]  # All raw issue data
    
    # Metadata
    extraction_date: str
    data_quality_score: float

class RawDataExtractor:
    def __init__(self, github_token: str):
        """Initialize with GitHub token"""
        self.github_token = github_token
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def extract_raw_repository_data(self, repo_url: str, manual_classifications: Dict[str, str]) -> Optional[RawRepositoryData]:
        """Extract raw repository data with individual issues"""
        try:
            owner, repo_name = self._parse_repo_url(repo_url)
            if not owner or not repo_name:
                print(f"Invalid repository URL: {repo_url}")
                return None
            
            print(f"Extracting raw data for {owner}/{repo_name}...")
            
            # Get basic repository info
            repo_info = self._get_repository_info(owner, repo_name)
            if not repo_info:
                return None
            
            # Get repository statistics
            repo_stats = self._get_repository_statistics(owner, repo_name)
            code_stats = self._get_code_statistics(owner, repo_name)
            contributor_stats = self._get_contributor_statistics(owner, repo_name)
            
            # Extract raw issues
            print(f"  Extracting raw issues...")
            raw_issues = self._extract_raw_issues(owner, repo_name, max_issues=500)
            
            # Basic calculations only
            basic_metrics = self._calculate_basic_metrics(repo_info, repo_stats, contributor_stats, code_stats, raw_issues)
            data_quality = self._assess_basic_quality(repo_info, repo_stats, code_stats, contributor_stats, len(raw_issues))
            
            # Construct raw data object
            repo_data = RawRepositoryData(
                # Identification
                repo_name=f"{owner}/{repo_name}",
                url=repo_url,
                classification_scope=manual_classifications.get('scope', 'Unknown'),
                classification_function=manual_classifications.get('function', 'Unknown'),
                
                # Raw metrics from GitHub API
                stars=repo_info.get('stargazers_count', 0),
                forks=repo_info.get('forks_count', 0),
                open_issues=repo_info.get('open_issues_count', 0),
                repo_age_months=basic_metrics.get('repo_age_months', 0),
                total_commits=repo_stats.get('commit_count', 0),
                total_releases=repo_stats.get('release_count', 0),
                total_contributors=contributor_stats.get('total_contributors', 0),
                active_contributors_3m=contributor_stats.get('active_contributors_3m', 0),
                total_loc=code_stats.get('total_loc', 0),
                language_breakdown=code_stats.get('language_breakdown', {}),
                file_count=code_stats.get('file_count', 0),
                
                # Basic issue aggregations
                total_issues_analyzed=len([i for i in raw_issues if not i.get('pull_request')]),
                closed_issues_count=len([i for i in raw_issues if i['state'] == 'closed' and not i.get('pull_request')]),
                
                # Your required calculated metrics
                issue_closure_rate=basic_metrics.get('issue_closure_rate', 0),
                commits_per_month_avg=basic_metrics.get('commits_per_month_avg', 0),
                maintenance_load=basic_metrics.get('maintenance_load', 0),
                contributor_vitality=basic_metrics.get('contributor_vitality', 0),
                avg_loc_per_file=basic_metrics.get('avg_loc_per_file', 0),
                
                # Raw individual issues (no analysis)
                raw_issues=[self._clean_raw_issue(issue) for issue in raw_issues if not issue.get('pull_request')],
                
                # Metadata
                extraction_date=datetime.now().isoformat(),
                data_quality_score=data_quality
            )
            
            print(f"Successfully extracted {len(repo_data.raw_issues)} raw issues")
            return repo_data
            
        except Exception as e:
            print(f"Error extracting data for {repo_url}: {e}")
            return None

    def _extract_raw_issues(self, owner: str, repo: str, max_issues: int = 500) -> List[Dict]:
        """Extract raw issues with no analysis - just GitHub API data"""
        raw_issues = []
        
        # Get both open and closed issues
        for state in ['open', 'closed']:
            issues_url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params = {'state': state, 'per_page': 100, 'sort': 'updated', 'direction': 'desc'}
            
            page = 1
            issues_collected = 0
            
            while issues_collected < max_issues // 2:  # Split between open/closed
                params['page'] = page
                issues = self._make_api_request(issues_url, params=params)
                
                if not issues or not isinstance(issues, list) or len(issues) == 0:
                    break
                
                for issue_data in issues:
                    if issues_collected >= max_issues // 2:
                        break
                    
                    # Store raw issue data as-is from GitHub API
                    raw_issues.append(issue_data)
                    issues_collected += 1
                
                if len(issues) < 100:
                    break
                page += 1
                
                if page > 5:  # Limit API calls
                    break
        
        return raw_issues

    def _clean_raw_issue(self, issue_data: Dict) -> Dict:
        """Clean and structure raw issue data - no analysis"""
        return {
            'number': issue_data['number'],
            'title': issue_data.get('title', ''),
            'body': issue_data.get('body', ''),
            'state': issue_data['state'],
            'created_at': issue_data['created_at'],
            'closed_at': issue_data.get('closed_at'),
            'updated_at': issue_data['updated_at'],
            'labels': [label['name'] for label in issue_data.get('labels', [])],
            'assignees': [assignee['login'] for assignee in issue_data.get('assignees', [])],
            'milestone': issue_data.get('milestone', {}).get('title') if issue_data.get('milestone') else None,
            'comments_count': issue_data['comments'],
            'reactions': issue_data.get('reactions', {}),
            'author': issue_data['user']['login'],
            'html_url': issue_data['html_url']
        }

    def _calculate_basic_metrics(self, repo_info: Dict, repo_stats: Dict, contributor_stats: Dict, 
                                code_stats: Dict, raw_issues: List[Dict]) -> Dict:
        """Calculate only the basic metrics required by your schema"""
        metrics = {}
        
        # Repository age
        created_at = repo_info.get('created_at', '')
        if created_at:
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age_days = (datetime.now(created_date.tzinfo) - created_date).days
            metrics['repo_age_months'] = round(age_days / 30.44, 2)
        else:
            metrics['repo_age_months'] = 0
        
        # Issue closure rate
        issues_only = [i for i in raw_issues if not i.get('pull_request')]
        closed_issues = len([i for i in issues_only if i['state'] == 'closed'])
        total_issues = len(issues_only)
        
        if total_issues > 0:
            metrics['issue_closure_rate'] = round(closed_issues / total_issues, 3)
        else:
            metrics['issue_closure_rate'] = 0
        
        # Commits per month average
        total_commits = repo_stats.get('commit_count', 0)
        repo_age_months = metrics['repo_age_months']
        
        if repo_age_months > 0:
            metrics['commits_per_month_avg'] = round(total_commits / repo_age_months, 2)
        else:
            metrics['commits_per_month_avg'] = 0
        
        # Maintenance load
        open_issues = repo_info.get('open_issues_count', 0)
        active_contributors = contributor_stats.get('active_contributors_3m', 0)
        
        if active_contributors > 0:
            metrics['maintenance_load'] = round(open_issues / active_contributors, 2)
        else:
            metrics['maintenance_load'] = float('inf') if open_issues > 0 else 0
        
        # Contributor vitality
        total_contributors = contributor_stats.get('total_contributors', 0)
        if total_contributors > 0:
            metrics['contributor_vitality'] = round(active_contributors / total_contributors, 3)
        else:
            metrics['contributor_vitality'] = 0
        
        # Average LOC per file
        total_loc = code_stats.get('total_loc', 0)
        file_count = code_stats.get('file_count', 0)
        
        if file_count > 0:
            metrics['avg_loc_per_file'] = round(total_loc / file_count, 1)
        else:
            metrics['avg_loc_per_file'] = 0
        
        return metrics

    # Include all the basic GitHub API methods (same as before)
    def _parse_repo_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        pattern = r'github\.com/([^/]+)/([^/\?#]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _get_repository_info(self, owner: str, repo: str) -> Optional[Dict]:
        url = f"{self.base_url}/repos/{owner}/{repo}"
        return self._make_api_request(url)

    def _get_repository_statistics(self, owner: str, repo: str) -> Dict:
        stats = {}
        
        # Get commit count
        commits_url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {'per_page': 1}
        response = self._make_api_request(commits_url, params=params)
        
        if response and isinstance(response, list):
            link_header = getattr(self, '_last_response_headers', {}).get('Link', '')
            if 'last' in link_header:
                last_match = re.search(r'page=(\d+)[^>]*>; rel="last"', link_header)
                if last_match:
                    stats['commit_count'] = int(last_match.group(1)) * 30
                else:
                    stats['commit_count'] = 100
            else:
                stats['commit_count'] = 30
        else:
            stats['commit_count'] = 0
        
        # Get releases
        releases_url = f"{self.base_url}/repos/{owner}/{repo}/releases"
        releases = self._make_api_request(releases_url)
        stats['release_count'] = len(releases) if releases and isinstance(releases, list) else 0
        
        return stats

    def _get_code_statistics(self, owner: str, repo: str) -> Dict:
        stats = {'total_loc': 0, 'language_breakdown': {}, 'file_count': 0}
        
        # Languages
        languages_url = f"{self.base_url}/repos/{owner}/{repo}/languages"
        languages = self._make_api_request(languages_url)
        
        if languages:
            total_bytes = sum(languages.values())
            if total_bytes > 0:
                stats['language_breakdown'] = {
                    lang: round((bytes_count / total_bytes) * 100, 1)
                    for lang, bytes_count in languages.items()
                }
                stats['total_loc'] = int(total_bytes * 0.01)
        
        # File count
        tree_url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/HEAD"
        params = {'recursive': '1'}
        tree = self._make_api_request(tree_url, params=params)
        
        if tree and 'tree' in tree:
            files = [item for item in tree['tree'] if item['type'] == 'blob']
            stats['file_count'] = len(files)
        
        return stats

    def _get_contributor_statistics(self, owner: str, repo: str) -> Dict:
        stats = {'total_contributors': 0, 'active_contributors_3m': 0}
        
        # Total contributors
        contributors_url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
        contributors = self._make_api_request(contributors_url)
        
        if contributors and isinstance(contributors, list):
            stats['total_contributors'] = len(contributors)
        
        # Active contributors (last 3 months)
        three_months_ago = datetime.now() - timedelta(days=90)
        commits_url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {'since': three_months_ago.isoformat() + 'Z', 'per_page': 100}
        
        active_contributors = set()
        page = 1
        
        while page <= 3:
            params['page'] = page
            commits = self._make_api_request(commits_url, params=params)
            
            if not commits or not isinstance(commits, list) or len(commits) == 0:
                break
            
            for commit in commits:
                if commit.get('author') and commit['author'].get('login'):
                    active_contributors.add(commit['author']['login'])
            
            if len(commits) < 100:
                break
            page += 1
        
        stats['active_contributors_3m'] = len(active_contributors)
        return stats

    def _assess_basic_quality(self, repo_info: Dict, repo_stats: Dict, code_stats: Dict, 
                            contributor_stats: Dict, issues_extracted: int) -> float:
        checks = [
            repo_info.get('stargazers_count') is not None,
            repo_info.get('forks_count') is not None,
            repo_info.get('created_at') is not None,
            repo_stats.get('commit_count', 0) > 0,
            code_stats.get('total_loc', 0) > 0,
            code_stats.get('file_count', 0) > 0,
            contributor_stats.get('total_contributors', 0) > 0,
            issues_extracted >= 0,
            len(code_stats.get('language_breakdown', {})) > 0
        ]
        
        return round(sum(checks) / len(checks), 3)

    def _make_api_request(self, url: str, params: Dict = None) -> Optional[Any]:
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self._last_response_headers = response.headers
            
            if response.status_code == 403:
                print("Rate limit reached, waiting...")
                time.sleep(60)
                response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                return None
            
            return response.json()
            
        except Exception as e:
            print(f"API request error: {e}")
            return None
        
        finally:
            time.sleep(0.1)

def process_repositories_raw_data(github_token: str, repositories: List[Dict], 
                                output_file: str = "raw_llm_dataset.json",
                                save_interval: int = 10):
    """Process repositories extracting raw data only"""
    extractor = RawDataExtractor(github_token)
    results = []
    failed_repos = []
    
    print(f"Starting raw data extraction of {len(repositories)} repositories...")
    print(f"No derived analysis - just raw GitHub data")
    print(f"Estimated time: {len(repositories) * 2} minutes")
    
    for i, repo_info in enumerate(repositories, 1):
        print(f"\n[{i}/{len(repositories)}] Processing: {repo_info['url']}")
        
        classifications = {
            'scope': repo_info.get('scope', 'Unknown'),
            'function': repo_info.get('function', 'Unknown')
        }
        
        repo_data = extractor.extract_raw_repository_data(
            repo_info['url'], 
            classifications
        )
        
        if repo_data:
            results.append(asdict(repo_data))
            print(f"Success - {len(repo_data.raw_issues)} issues extracted")
        else:
            failed_repos.append({
                'url': repo_info['url'],
                'attempted_at': datetime.now().isoformat()
            })
            print("Failed")
        
        # Save progress
        if i % save_interval == 0 or i == len(repositories):
            dataset = {
                'metadata': {
                    'extraction_type': 'raw_data_only',
                    'extraction_date': datetime.now().isoformat(),
                    'total_repositories_attempted': i,
                    'successful_extractions': len(results),
                    'failed_extractions': len(failed_repos),
                    'success_rate_percent': round((len(results) / i) * 100, 1),
                    'total_raw_issues_extracted': sum([r['total_issues_analyzed'] for r in results])
                },
                'schema_info': {
                    'description': 'Raw GitHub data with individual issues - no derived technical debt analysis',
                    'user_analysis_note': 'User will perform their own technical debt classifications and analysis'
                },
                'repositories': results,
                'failed_repositories': failed_repos
            }
            
            temp_file = f"{output_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            import os
            os.rename(temp_file, output_file)
            
            total_issues = dataset['metadata']['total_raw_issues_extracted']
            print(f"Progress saved: {len(results)} repos, {total_issues} raw issues")
    
    print(f"\nRaw data extraction complete!")
    return results

def main():
    github_token = "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"  # Replace with your token
    
    print("Raw LLM Repository Data Extraction")
    print("Extracts raw GitHub data only - no derived analysis")
    print("=" * 60)
    
    # Load dataset
    try:
        with open('complete_158_repositories_clean.json', 'r') as f:
            data = json.load(f)
            repositories = data['repositories']
        
        print(f"Loaded {len(repositories)} repositories")
        
    except FileNotFoundError:
        print("Dataset file not found.")
        return
    
    # Extract raw data
    results = process_repositories_raw_data(
        github_token=github_token,
        repositories=repositories,
        output_file="raw_llm_dataset.json",
        save_interval=10
    )
    
    if results:
        total_issues = sum([r['total_issues_analyzed'] for r in results])
        print(f"\nFinal Statistics:")
        print(f"Repositories: {len(results)}")
        print(f"Raw issues extracted: {total_issues}")
        print(f"Average issues per repo: {total_issues / len(results):.1f}")
        print(f"\nRaw dataset ready for your technical debt analysis!")

if __name__ == "__main__":
    main()