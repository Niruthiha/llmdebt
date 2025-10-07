
#!/usr/bin/env python3
"""
General & Agent Chatbots Discovery
Target only General chatbots and Agentic chatbots (no RAG)

Requirements:
pip install requests pandas python-dotenv ratelimit tqdm
"""

import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityCriteria:
    """Relaxed quality criteria for chatbot applications"""
    min_stars: int = 10            # Lowered for applications
    min_age_days: int = 180        # 6 months instead of 1 year
    min_commits: int = 5           # Lowered from 10
    max_days_since_push: int = 180 # More flexible
    min_contributors: int = 2      # Lowered from 3
    post_ai_boom_date: str = "2022-11-01"

# Focused search queries for General and Agent chatbots only
CHATBOT_QUERIES = [
    # General chatbot applications (basic conversation)
    'chatbot language:Python stars:>=10 pushed:>=2022-11-01',
    'chat-ui language:Python stars:>=10 pushed:>=2022-11-01',
    'chatbot-ui language:Python stars:>=10 pushed:>=2022-11-01',
    'chat-interface language:Python stars:>=10 pushed:>=2022-11-01',
    'gpt-chat language:Python stars:>=10 pushed:>=2022-11-01',
    'openai-chat language:Python stars:>=10 pushed:>=2022-11-01',
    'claude-chat language:Python stars:>=10 pushed:>=2022-11-01',
    
    # Text generation and conversation apps
    'text-generator language:Python stars:>=10 pushed:>=2022-11-01',
    'conversation language:Python stars:>=10 pushed:>=2022-11-01',
    'ai-writer language:Python stars:>=10 pushed:>=2022-11-01',
    'prompt-playground language:Python stars:>=10 pushed:>=2022-11-01',
    'llm-playground language:Python stars:>=10 pushed:>=2022-11-01',
    
    # Agent and autonomous applications
    'ai-assistant language:Python stars:>=10 pushed:>=2022-11-01',
    'personal-assistant language:Python stars:>=10 pushed:>=2022-11-01',
    'virtual-assistant language:Python stars:>=10 pushed:>=2022-11-01',
    'auto-gpt language:Python stars:>=10 pushed:>=2022-11-01',
    'baby-agi language:Python stars:>=10 pushed:>=2022-11-01',
    'autonomous-agent language:Python stars:>=10 pushed:>=2022-11-01',
    'task-automation language:Python stars:>=10 pushed:>=2022-11-01',
    'workflow-automation language:Python stars:>=10 pushed:>=2022-11-01',
    
    # UI framework applications
    'streamlit language:Python chatbot stars:>=10 pushed:>=2022-11-01',
    'gradio language:Python chat stars:>=10 pushed:>=2022-11-01',
    'chainlit language:Python stars:>=10 pushed:>=2022-11-01',
    
    # API key configuration (strong end-user app indicators)
    '"api key" language:Python stars:>=10 pushed:>=2022-11-01',
    '"openai api" language:Python stars:>=10 pushed:>=2022-11-01',
    '"your api key" language:Python stars:>=10 pushed:>=2022-11-01',
    '"set api key" language:Python stars:>=10 pushed:>=2022-11-01'
]

# Classification keywords (General vs Agent only)
CHATBOT_CLASSIFICATION = {
    "Agent": {
        "strong": ["auto-gpt", "baby-agi", "autonomous-agent", "ai-agent", "task-automation",
                  "personal-assistant", "virtual-assistant", "workflow-automation"],
        "medium": ["assistant", "autonomous", "automation", "workflow", "task", "planning"],
        "weak": ["agent", "execute", "action", "tool", "goal", "objective"]
    },
    "General": {
        "strong": ["chatbot-ui", "chat-interface", "simple-chat", "basic-chat", "gpt-chat",
                  "text-generator", "conversation", "llm-playground", "prompt-playground"],
        "medium": ["chatbot", "chat", "conversation", "dialogue", "text-generation"],
        "weak": ["talk", "generate", "prompt", "completion", "ai-chat"]
    }
}

# Strong exclusion criteria (frameworks/tools)
FRAMEWORK_EXCLUSION_TERMS = [
    "framework", "library", "toolkit", "sdk", "engine", "core", "client", "wrapper",
    "api", "package", "module", "plugin", "driver", "connector", "server", "backend",
    "for developers", "development tool", "programming", "build apps", "create apps"
]

# Strong inclusion criteria (end-user applications)
APPLICATION_INCLUSION_TERMS = [
    # User interface
    "web app", "desktop app", "user interface", "gui", "frontend", "webapp",
    "easy to use", "no-code", "simple setup", "user-friendly", "install and run",
    
    # API configuration (strong indicators)
    "api key", "openai_api_key", "your api key", "set your api key", "enter your api key",
    "configure api", "api_key =", "client = openai", "anthropic_api_key", "claude_api_key",
    
    # User interaction
    "upload file", "type message", "chat with", "ask question", "getting started",
    "step 1", "step 2", "how to use", "tutorial", "setup", "configuration"
]

# RAG exclusion terms (remove RAG-focused applications)
RAG_EXCLUSION_TERMS = [
    "document", "pdf", "file", "upload", "knowledge", "retrieval", "search",
    "vector", "embedding", "database", "index", "rag", "ask-pdf", "chat-pdf",
    "document-chat", "knowledge-base", "semantic-search"
]

class GeneralAgentChatbotDiscovery:
    def __init__(self, github_token: str, criteria: QualityCriteria = None):
        if not github_token:
            raise ValueError("GitHub token is required")
        
        self.token = github_token
        self.headers = {"Authorization": f"token {github_token}"}
        self.base_url = "https://api.github.com"
        self.criteria = criteria or QualityCriteria()
        
        # Storage
        self.discovered_repos: Dict[int, Dict] = {}
        self.search_stats = {
            "total_queries": 0,
            "total_found": 0,
            "duplicates_removed": 0,
            "quality_passed": 0,
            "quality_failed": 0,
            "applications_identified": 0,
            "frameworks_excluded": 0,
            "rag_excluded": 0
        }

    @sleep_and_retry
    @limits(calls=30, period=60)
    def search_repositories(self, query: str, max_results: int = 300) -> List[Dict]:
        """Search repositories with rate limiting"""
        repos = []
        page = 1
        
        while len(repos) < max_results:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc", 
                "per_page": min(100, max_results - len(repos)),
                "page": page
            }
            
            try:
                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if not data.get('items'):
                        break
                    
                    repos.extend(data['items'])
                    page += 1
                    
                    if len(data['items']) < params['per_page']:
                        break
                        
                elif response.status_code == 403:
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(60)
                    continue
                else:
                    logger.error(f"Error {response.status_code}: {response.text}")
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
        
        return repos

    @sleep_and_retry
    @limits(calls=5000, period=3600)
    def get_repository_readme(self, full_name: str) -> str:
        """Get repository README content"""
        try:
            url = f"{self.base_url}/repos/{full_name}/readme"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                readme_data = response.json()
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                return readme_content
            else:
                return ""
        except Exception as e:
            logger.debug(f"Could not get README for {full_name}: {e}")
            return ""

    @sleep_and_retry
    @limits(calls=5000, period=3600)
    def get_contributors_and_commits(self, full_name: str) -> tuple:
        """Get contributor and commit counts"""
        try:
            # Contributors
            contributors_url = f"{self.base_url}/repos/{full_name}/contributors"
            contrib_response = requests.get(contributors_url, headers=self.headers, timeout=30)
            contributor_count = len(contrib_response.json()) if contrib_response.status_code == 200 else 0
            
            # Commits
            commits_url = f"{self.base_url}/repos/{full_name}/commits"
            commit_response = requests.get(commits_url, headers=self.headers, timeout=30)
            commit_count = len(commit_response.json()) if commit_response.status_code == 200 else 0
            
            return contributor_count, commit_count
            
        except requests.RequestException:
            return 0, 0

    def is_end_user_application(self, repo: Dict) -> Tuple[bool, str, float]:
        """Check if repository is an end-user application (not framework/RAG)"""
        name = str(repo.get('name', '') or '').lower()
        description = str(repo.get('description', '') or '').lower()
        readme = str(repo.get('readme_content', '') or '').lower()
        
        all_text = f"{name} {description} {readme}"
        
        # Check for RAG exclusions first
        rag_score = sum(1 for term in RAG_EXCLUSION_TERMS if term in all_text)
        if rag_score >= 3:  # Strong RAG indicators
            return False, f"RAG-focused ({rag_score} RAG terms)", 0.8
        
        # Check for framework exclusions
        framework_score = sum(1 for term in FRAMEWORK_EXCLUSION_TERMS if term in all_text)
        
        # Check for application inclusions
        app_score = sum(1 for term in APPLICATION_INCLUSION_TERMS if term in all_text)
        
        # Special penalties for framework indicators in name
        if any(term in name for term in ['framework', 'library', 'sdk', 'toolkit', 'core']):
            framework_score += 5
        
        # Special bonuses for app indicators in name
        if any(term in name for term in ['ui', 'app', 'chat', 'bot', 'interface']):
            app_score += 3
        
        # Decision logic
        if framework_score > app_score + 2:
            confidence = framework_score / (framework_score + app_score + 1)
            return False, f"Framework/SDK ({framework_score} indicators)", confidence
        elif app_score > 0:
            confidence = app_score / (framework_score + app_score + 1)
            return True, f"End-user app ({app_score} indicators)", confidence
        else:
            return False, "Unclear purpose", 0.1

    def classify_chatbot_type(self, repo: Dict) -> Tuple[str, float]:
        """Classify as General or Agent chatbot"""
        name = str(repo.get('name', '') or '').lower()
        description = str(repo.get('description', '') or '').lower()
        readme = str(repo.get('readme_content', '') or '').lower()
        
        all_text = f"{name} {description} {readme}"
        
        # Score Agent vs General
        agent_score = 0
        general_score = 0
        
        # Agent scoring
        for keyword in CHATBOT_CLASSIFICATION["Agent"]["strong"]:
            if keyword in name:
                agent_score += 10
            elif keyword in description:
                agent_score += 5
            else:
                agent_score += all_text.count(keyword) * 3
        
        for keyword in CHATBOT_CLASSIFICATION["Agent"]["medium"]:
            agent_score += all_text.count(keyword) * 2
        
        for keyword in CHATBOT_CLASSIFICATION["Agent"]["weak"]:
            agent_score += all_text.count(keyword)
        
        # General scoring
        for keyword in CHATBOT_CLASSIFICATION["General"]["strong"]:
            if keyword in name:
                general_score += 10
            elif keyword in description:
                general_score += 5
            else:
                general_score += all_text.count(keyword) * 3
        
        for keyword in CHATBOT_CLASSIFICATION["General"]["medium"]:
            general_score += all_text.count(keyword) * 2
        
        for keyword in CHATBOT_CLASSIFICATION["General"]["weak"]:
            general_score += all_text.count(keyword)
        
        # Classification decision
        if agent_score > general_score + 2:  # Clear agent
            confidence = min(agent_score / (agent_score + general_score + 1), 1.0)
            return "Agent", confidence
        elif general_score > 0:  # Some general indicators
            confidence = min(general_score / (agent_score + general_score + 1), 1.0)
            return "General", confidence
        else:
            return "General", 0.3  # Default to general

    def check_quality_criteria(self, repo: Dict) -> bool:
        """Apply relaxed quality criteria"""
        try:
            if not repo.get('full_name') or repo.get('fork', False):
                return False
            
            if repo.get('stargazers_count', 0) < self.criteria.min_stars:
                return False
            
            # Age check (6 months)
            created_date = datetime.strptime(repo['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            age_days = (datetime.now() - created_date).days
            if age_days < self.criteria.min_age_days:
                return False
            
            # Push recency check (6 months)
            if repo.get('pushed_at'):
                push_date = datetime.strptime(repo['pushed_at'], "%Y-%m-%dT%H:%M:%SZ")
                days_since_push = (datetime.now() - push_date).days
                if days_since_push > self.criteria.max_days_since_push:
                    return False
            else:
                return False
            
            # Get contributors and commits
            contributor_count, commit_count = self.get_contributors_and_commits(repo['full_name'])
            
            if contributor_count <= self.criteria.min_contributors:
                return False
            
            if commit_count < self.criteria.min_commits:
                return False
            
            # Store metadata
            repo['contributor_count'] = contributor_count
            repo['commit_count_estimate'] = commit_count
            repo['age_days'] = age_days
            repo['days_since_push'] = days_since_push
            
            return True
            
        except Exception as e:
            logger.error(f"Quality check failed for {repo.get('full_name', 'unknown')}: {e}")
            return False

    def process_repository(self, repo: Dict) -> Dict:
        """Process and classify a repository"""
        try:
            # Get README content
            readme_content = self.get_repository_readme(repo['full_name'])
            repo['readme_content'] = readme_content
            
            # Check if it's an end-user application (not framework/RAG)
            is_app, app_reason, app_confidence = self.is_end_user_application(repo)
            
            if not is_app:
                return None  # Skip non-applications
            
            # Classify chatbot type (General vs Agent)
            chatbot_type, type_confidence = self.classify_chatbot_type(repo)
            
            # Create processed repository
            processed_repo = {
                **repo,
                'is_application': True,
                'application_reason': app_reason,
                'application_confidence': app_confidence,
                'chatbot_type': chatbot_type,
                'type_confidence': type_confidence,
                'overall_confidence': (app_confidence + type_confidence) / 2,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return processed_repo
            
        except Exception as e:
            logger.error(f"Processing failed for {repo.get('full_name', 'unknown')}: {e}")
            return None

    def discover_and_classify_chatbots(self) -> Dict[str, List[Dict]]:
        """Main discovery and classification method"""
        logger.info("Starting General & Agent chatbot discovery...")
        
        all_applications = []
        
        for i, query in enumerate(CHATBOT_QUERIES):
            logger.info(f"Query {i+1}/{len(CHATBOT_QUERIES)}: {query}")
            
            repos = self.search_repositories(query, max_results=200)
            self.search_stats["total_found"] += len(repos)
            
            if not repos:
                continue
            
            logger.info(f"Found {len(repos)} repos, filtering and classifying...")
            
            for repo in tqdm(repos, desc=f"Processing {i+1}/{len(CHATBOT_QUERIES)}"):
                repo_id = repo['id']
                
                # Skip duplicates
                if repo_id in self.discovered_repos:
                    self.search_stats["duplicates_removed"] += 1
                    continue
                
                # Apply quality criteria
                if not self.check_quality_criteria(repo):
                    self.search_stats["quality_failed"] += 1
                    continue
                
                self.search_stats["quality_passed"] += 1
                
                # Process and classify
                processed_repo = self.process_repository(repo)
                
                if processed_repo:
                    # Check if it was excluded due to RAG focus
                    if "RAG-focused" in processed_repo.get('application_reason', ''):
                        self.search_stats["rag_excluded"] += 1
                        continue
                    
                    all_applications.append(processed_repo)
                    self.discovered_repos[repo_id] = processed_repo
                    self.search_stats["applications_identified"] += 1
                else:
                    self.search_stats["frameworks_excluded"] += 1
            
            logger.info(f"Query {i+1}: {self.search_stats['applications_identified']} total applications")
            time.sleep(2)
        
        # Group by chatbot type
        chatbot_types = {"General": [], "Agent": []}
        
        for app in all_applications:
            chatbot_type = app['chatbot_type']
            if chatbot_type in chatbot_types:
                chatbot_types[chatbot_type].append(app)
        
        # Sort by confidence and stars
        for chatbot_type in chatbot_types:
            chatbot_types[chatbot_type].sort(
                key=lambda x: (x['overall_confidence'], x['stargazers_count']),
                reverse=True
            )
        
        return chatbot_types

    def save_results(self, chatbot_types: Dict[str, List[Dict]], output_dir: str = "output"):
        """Save results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        all_chatbots = []
        for chatbot_type, repos in chatbot_types.items():
            all_chatbots.extend(repos)
        
        if not all_chatbots:
            logger.warning("No chatbot applications found")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_chatbots)
        
        columns = [
            'id', 'name', 'full_name', 'description', 'html_url',
            'stargazers_count', 'contributor_count', 'commit_count_estimate',
            'age_days', 'days_since_push', 'chatbot_type', 'type_confidence',
            'application_confidence', 'overall_confidence', 'application_reason'
        ]
        
        available_columns = [col for col in columns if col in df.columns]
        df_clean = df[available_columns]
        df_clean = df_clean.sort_values(['chatbot_type', 'overall_confidence', 'stargazers_count'], 
                                       ascending=[True, False, False])
        
        df_clean['selected'] = ''
        df_clean['notes'] = ''
        
        df_clean.to_csv(f"{output_dir}/general_agent_chatbots.csv", index=False)
        
        # Selection candidates
        selection_data = []
        target_per_type = 30  # Top 30 per type
        
        for chatbot_type, repos in chatbot_types.items():
            for rank, repo in enumerate(repos[:target_per_type], 1):
                selection_data.append({
                    'rank': rank,
                    'chatbot_type': chatbot_type,
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'description': repo.get('description', ''),
                    'stars': repo['stargazers_count'],
                    'confidence': round(repo['overall_confidence'], 3),
                    'type_confidence': round(repo['type_confidence'], 3),
                    'html_url': repo['html_url'],
                    'selected': '',
                    'notes': ''
                })
        
        df_selection = pd.DataFrame(selection_data)
        df_selection.to_csv(f"{output_dir}/chatbot_selection_final.csv", index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("GENERAL & AGENT CHATBOT DISCOVERY SUMMARY")
        print("="*70)
        print("Target: End-user chatbots only (General + Agent types)")
        print(f"Total applications found: {len(all_chatbots)}")
        print(f"Frameworks excluded: {self.search_stats['frameworks_excluded']}")
        print(f"RAG applications excluded: {self.search_stats['rag_excluded']}")
        print(f"Applications identified: {self.search_stats['applications_identified']}")
        print("-"*70)
        print("Chatbot types found:")
        
        for chatbot_type, repos in chatbot_types.items():
            high_conf = len([r for r in repos if r['overall_confidence'] >= 0.6])
            print(f"  {chatbot_type:8}: {len(repos):3d} total, {high_conf:3d} high-confidence")
        
        print("="*70)
        print("\nOutput files:")
        print(f"  📊 general_agent_chatbots.csv - All {len(all_chatbots)} chatbot apps")
        print(f"  🎯 chatbot_selection_final.csv - Top 30 per type for selection")
        print("="*70)
        print("\nFINAL SELECTION:")
        print("1. Review chatbot_selection_final.csv")
        print("2. Select 30 repositories per type:")
        print("   - General: Basic chat, text generation, conversation")
        print("   - Agent: Autonomous assistants, task automation")
        print("3. Mark 'selected' = 'Yes' for chosen repositories")
        print("4. Final sample: 60 chatbot applications")
        print("="*70)

def main():
    """Main execution function"""
    if not GITHUB_TOKEN:
        print("Error: Please set GITHUB_TOKEN environment variable")
        return
    
    print("🤖 General & Agent Chatbot Discovery")
    print("Target: End-user chatbot applications only")
    print("Categories: General chatbots + Agent chatbots (NO RAG)")
    print("Estimated time: 45-60 minutes")
    print()
    print("Relaxed Quality Criteria:")
    criteria = QualityCriteria()
    print(f"  Stars: >={criteria.min_stars}")
    print(f"  Age: >={criteria.min_age_days} days")
    print(f"  Commits: >={criteria.min_commits}")
    print(f"  Last push: <={criteria.max_days_since_push} days ago")
    print(f"  Contributors: >{criteria.min_contributors}")
    print()
    
    # Initialize
    discovery = GeneralAgentChatbotDiscovery(GITHUB_TOKEN, criteria)
    
    try:
        # Run discovery
        chatbot_types = discovery.discover_and_classify_chatbots()
        
        # Save results
        discovery.save_results(chatbot_types)
        
        print("\n✅ Discovery completed!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        print(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
