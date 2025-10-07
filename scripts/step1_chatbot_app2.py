#!/usr/bin/env python3
"""
User-Facing Chatbots Discovery
Target ONLY end-user chatbot applications (no frameworks/SDKs/libraries)

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
    """Very relaxed criteria for user-facing applications"""
    min_stars: int = 5             # Very low for small projects
    min_age_days: int = 120        # 4 months
    min_commits: int = 3           # Very low
    max_days_since_push: int = 365 # 1 year (very flexible)
    min_contributors: int = 1      # Even solo projects
    post_ai_boom_date: str = "2022-11-01"

# Highly specific search queries for AGENTIC end-user applications
AGENTIC_USER_APPLICATIONS = [
    # LangGraph-based user applications
    'langgraph app language:Python stars:>=5 pushed:>=2022-11-01',
    'langgraph streamlit language:Python stars:>=5 pushed:>=2022-11-01',
    'langgraph assistant language:Python stars:>=5 pushed:>=2022-11-01',
    'langgraph chatbot language:Python stars:>=5 pushed:>=2022-11-01',
    
    # OpenAI Agents user applications
    '"openai agent" language:Python stars:>=5 pushed:>=2022-11-01',
    'openai assistant language:Python stars:>=5 pushed:>=2022-11-01',
    'openai automation language:Python stars:>=5 pushed:>=2022-11-01',
    'openai swarm language:Python stars:>=5 pushed:>=2022-11-01',
    
    # Task automation applications for users
    'task automation app language:Python stars:>=5 pushed:>=2022-11-01',
    'workflow automation language:Python stars:>=5 pushed:>=2022-11-01',
    'ai task manager language:Python stars:>=5 pushed:>=2022-11-01',
    'personal task assistant language:Python stars:>=5 pushed:>=2022-11-01',
    
    # AutoGPT-style applications
    'auto-gpt clone language:Python stars:>=5 pushed:>=2022-11-01',
    'autogpt app language:Python stars:>=5 pushed:>=2022-11-01',
    'autonomous agent app language:Python stars:>=5 pushed:>=2022-11-01',
    'baby-agi clone language:Python stars:>=5 pushed:>=2022-11-01',
    
    # Agent applications with UI
    'agent streamlit language:Python stars:>=5 pushed:>=2022-11-01',
    'agent gradio language:Python stars:>=5 pushed:>=2022-11-01',
    'agent interface language:Python stars:>=5 pushed:>=2022-11-01',
    'agent app language:Python stars:>=5 pushed:>=2022-11-01',
    
    # Personal assistant applications
    'personal assistant app language:Python stars:>=5 pushed:>=2022-11-01',
    'ai assistant app language:Python stars:>=5 pushed:>=2022-11-01',
    'virtual assistant app language:Python stars:>=5 pushed:>=2022-11-01',
    'voice assistant language:Python stars:>=5 pushed:>=2022-11-01',
    
    # Productivity and automation apps
    'ai productivity language:Python stars:>=5 pushed:>=2022-11-01',
    'automation app language:Python stars:>=5 pushed:>=2022-11-01',
    'workflow app language:Python stars:>=5 pushed:>=2022-11-01',
    'task planner language:Python stars:>=5 pushed:>=2022-11-01',
    
    # User interface terms with agent functionality
    'agentic ui language:Python stars:>=5 pushed:>=2022-11-01',
    'autonomous ui language:Python stars:>=5 pushed:>=2022-11-01',
    'agent dashboard language:Python stars:>=5 pushed:>=2022-11-01',
    
    # General chatbot applications (non-agent)
    'chatbot streamlit language:Python stars:>=5 pushed:>=2022-11-01',
    'chat streamlit language:Python stars:>=5 pushed:>=2022-11-01',
    'chatbot gradio language:Python stars:>=5 pushed:>=2022-11-01',
    'chat gradio language:Python stars:>=5 pushed:>=2022-11-01',
    'simple chatbot language:Python stars:>=5 pushed:>=2022-11-01',
    'basic chatbot language:Python stars:>=5 pushed:>=2022-11-01',
    
    # Text generation applications
    'text generator app language:Python stars:>=5 pushed:>=2022-11-01',
    'ai writer app language:Python stars:>=5 pushed:>=2022-11-01',
    'story generator language:Python stars:>=5 pushed:>=2022-11-01',
    'content generator language:Python stars:>=5 pushed:>=2022-11-01'
]

# Very specific application indicators
STRONG_APPLICATION_INDICATORS = [
    # UI frameworks (almost always end-user apps)
    "streamlit", "gradio", "chainlit", "flask", "fastapi", "django",
    "tkinter", "pygame", "kivy", "qt", "gui", "web app", "webapp",
    
    # User interaction patterns
    "run app", "python app.py", "streamlit run", "localhost", "127.0.0.1",
    "open browser", "navigate to", "click", "upload", "download",
    
    # User configuration
    "enter your", "your api key", "paste your", "configure", "setup",
    "api key here", "your openai", "your anthropic", "your claude",
    
    # Application deployment
    "deploy", "hosted", "live demo", "try it", "demo", "playground",
    "web interface", "user interface", "frontend", "client"
]

# Strong framework/library exclusions
FRAMEWORK_EXCLUSIONS = [
    "framework", "library", "toolkit", "sdk", "engine", "core", "base",
    "client library", "python package", "pip install", "import", "module",
    "api wrapper", "api client", "for developers", "development",
    "building", "create", "build apps", "developer tool"
]

# Agent vs General classification keywords
AGENT_INDICATORS = [
    # LangGraph usage (strong agent signal)
    "langgraph", "lang_graph", "langgraph agent", "langgraph workflow",
    
    # OpenAI Agents/Swarm usage
    "openai agent", "openai swarm", "swarm agent", "agents openai",
    
    # Autonomous behavior terms
    "autonomous", "auto-gpt", "baby-agi", "task automation", "workflow automation",
    "goal-oriented", "objective", "planning", "decision making", "execute tasks",
    
    # Assistant terms (agents)
    "personal assistant", "ai assistant", "virtual assistant", "task assistant",
    "productivity assistant", "work assistant", "scheduling assistant",
    
    # Agent functionality
    "tool calling", "function calling", "multi-step", "chain of thought",
    "reasoning", "problem solving", "task execution", "action selection"
]

GENERAL_INDICATORS = [
    # Basic conversation
    "chat", "conversation", "dialogue", "talk", "simple chat", "basic chat",
    "text generation", "story generation", "content generation",
    
    # Interface terms (general)
    "prompt", "completion", "playground", "interface", "ui", "frontend",
    "chatbot ui", "chat interface", "conversation ui",
    
    # Clone/replica applications
    "chatgpt clone", "claude clone", "gemini clone", "openai clone",
    "gpt replica", "ai replica"
]

class UserFacingChatbotDiscovery:
    def __init__(self, github_token: str, criteria: QualityCriteria = None):
        if not github_token:
            raise ValueError("GitHub token is required")
        
        self.token = github_token
        self.headers = {"Authorization": f"token {github_token}"}
        self.base_url = "https://api.github.com"
        self.criteria = criteria or QualityCriteria()
        
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
                    logger.error(f"Error {response.status_code}")
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
        
        return repos

    @sleep_and_retry
    @limits(calls=5000, period=3600)
    def get_repository_readme(self, full_name: str) -> str:
        """Get README content"""
        try:
            url = f"{self.base_url}/repos/{full_name}/readme"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                readme_data = response.json()
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                return readme_content
            return ""
        except Exception:
            return ""

    @sleep_and_retry
    @limits(calls=5000, period=3600)
    def get_contributors_and_commits(self, full_name: str) -> tuple:
        """Get basic repo stats"""
        try:
            contributors_url = f"{self.base_url}/repos/{full_name}/contributors"
            contrib_response = requests.get(contributors_url, headers=self.headers, timeout=30)
            contributor_count = len(contrib_response.json()) if contrib_response.status_code == 200 else 0
            
            commits_url = f"{self.base_url}/repos/{full_name}/commits"
            commit_response = requests.get(commits_url, headers=self.headers, timeout=30)
            commit_count = len(commit_response.json()) if commit_response.status_code == 200 else 0
            
            return contributor_count, commit_count
        except:
            return 0, 0

    def is_user_facing_application(self, repo: Dict) -> Tuple[bool, str, float]:
        """Enhanced filtering for agentic user applications"""
        name = str(repo.get('name', '') or '').lower()
        description = str(repo.get('description', '') or '').lower()
        readme = str(repo.get('readme_content', '') or '').lower()
        
        all_text = f"{name} {description} {readme}"
        
        # HARD EXCLUSIONS - backend frameworks and libraries
        backend_exclusions = [
            "framework", "library", "toolkit", "sdk", "engine", "package", "module",
            "pip install", "import", "api wrapper", "client library", "core library",
            "for developers", "development", "building", "create apps", "build agents",
            "backend", "server", "microservice", "infrastructure", "platform"
        ]
        
        for exclusion in backend_exclusions:
            if exclusion in name or exclusion in description:
                return False, f"Backend/Framework: {exclusion}", 0.9
        
        # RAG exclusions (you have enough)
        rag_terms = ["document", "pdf", "file upload", "knowledge base", "retrieval", "rag", "vector database"]
        rag_count = sum(1 for term in rag_terms if term in all_text)
        if rag_count >= 2:
            return False, f"RAG-focused ({rag_count} terms)", 0.8
        
        # STRONG USER APPLICATION INDICATORS
        app_score = 0
        app_reasons = []
        
        # LangGraph user applications (not framework itself)
        langgraph_user_patterns = [
            "langgraph app", "langgraph streamlit", "langgraph gradio", 
            "langgraph assistant", "langgraph chatbot", "using langgraph"
        ]
        for pattern in langgraph_user_patterns:
            if pattern in all_text:
                app_score += 6
                app_reasons.append("langgraph_user_app")
                break
        
        # OpenAI Agents user applications
        openai_agent_patterns = [
            "openai agent", "openai swarm", "using openai agents", 
            "openai assistant", "swarm agent", "agents with openai"
        ]
        for pattern in openai_agent_patterns:
            if pattern in all_text:
                app_score += 6
                app_reasons.append("openai_agent_app")
                break
        
        # UI framework usage (strong user app signal)
        ui_frameworks = ["streamlit", "gradio", "chainlit", "flask", "fastapi", "django"]
        for framework in ui_frameworks:
            if framework in all_text:
                app_score += 4
                app_reasons.append(f"uses_{framework}")
        
        # User interaction setup
        user_setup_patterns = [
            "streamlit run", "python app.py", "run the app", "localhost",
            "open browser", "navigate to", "127.0.0.1", "start the application"
        ]
        for pattern in user_setup_patterns:
            if pattern in all_text:
                app_score += 4
                app_reasons.append("user_setup")
                break
        
        # API key configuration for users
        api_config_patterns = [
            "enter your api key", "your openai api key", "set your api key",
            "api_key =", "configure your", "add your api key"
        ]
        for pattern in api_config_patterns:
            if pattern in all_text:
                app_score += 5
                app_reasons.append("user_api_config")
                break
        
        # User interface indicators
        ui_terms = ["app", "ui", "interface", "frontend", "web app", "desktop app", "gui"]
        ui_in_name = sum(1 for term in ui_terms if term in name)
        ui_in_desc = sum(1 for term in ui_terms if term in description)
        
        if ui_in_name > 0:
            app_score += ui_in_name * 3
            app_reasons.append("ui_in_name")
        if ui_in_desc > 0:
            app_score += ui_in_desc * 2
            app_reasons.append("ui_in_description")
        
        # Task automation for users (not backend automation)
        user_automation_patterns = [
            "personal task", "daily task", "automate your", "task manager app",
            "productivity app", "workflow app", "schedule tasks", "manage tasks"
        ]
        automation_count = sum(1 for pattern in user_automation_patterns if pattern in all_text)
        if automation_count > 0:
            app_score += automation_count * 3
            app_reasons.append("user_automation")
        
        # Decision: require strong evidence for user-facing application
        if app_score >= 6:  # Higher threshold for better filtering
            confidence = min(app_score / 20, 1.0)
            return True, f"User app: {', '.join(app_reasons[:3])}", confidence
        else:
            return False, f"Not user-facing (score: {app_score})", 0.3

    def classify_chatbot_type(self, repo: Dict) -> Tuple[str, float]:
        """Classify as General or Agent"""
        name = str(repo.get('name', '') or '').lower()
        description = str(repo.get('description', '') or '').lower()
        readme = str(repo.get('readme_content', '') or '').lower()
        
        all_text = f"{name} {description} {readme}"
        
        # Score Agent indicators
        agent_score = 0
        for indicator in AGENT_INDICATORS:
            if indicator in name:
                agent_score += 5  # Strong signal in name
            elif indicator in description:
                agent_score += 3  # Medium signal in description
            else:
                agent_score += all_text.count(indicator)
        
        # Score General indicators
        general_score = 0
        for indicator in GENERAL_INDICATORS:
            if indicator in name:
                general_score += 5
            elif indicator in description:
                general_score += 3
            else:
                general_score += all_text.count(indicator)
        
        # Classification with clear priority
        if agent_score > general_score + 2:
            confidence = min(agent_score / (agent_score + general_score + 1), 1.0)
            return "Agent", confidence
        else:
            confidence = min(general_score / (agent_score + general_score + 1), 1.0)
            return "General", max(confidence, 0.3)  # Default to General

    def check_quality_criteria(self, repo: Dict) -> bool:
        """Apply very relaxed quality criteria for applications"""
        try:
            if not repo.get('full_name') or repo.get('fork', False):
                return False
            
            if repo.get('stargazers_count', 0) < self.criteria.min_stars:
                return False
            
            # Age check (4 months)
            created_date = datetime.strptime(repo['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            age_days = (datetime.now() - created_date).days
            if age_days < self.criteria.min_age_days:
                return False
            
            # Push check (1 year flexibility)
            if repo.get('pushed_at'):
                push_date = datetime.strptime(repo['pushed_at'], "%Y-%m-%dT%H:%M:%SZ")
                days_since_push = (datetime.now() - push_date).days
                if days_since_push > self.criteria.max_days_since_push:
                    return False
            else:
                return False
            
            # Very relaxed contributor/commit checks
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
            logger.error(f"Quality check failed: {e}")
            return False

    def discover_user_chatbots(self) -> Dict[str, List[Dict]]:
        """Main discovery method targeting user-facing chatbots only"""
        logger.info("Starting user-facing chatbot discovery...")
        logger.info("Target: General chatbots + Agent chatbots (NO frameworks, NO RAG)")
        
        all_applications = []
        
        for i, query in enumerate(AGENTIC_USER_APPLICATIONS):
            logger.info(f"Query {i+1}/{len(AGENTIC_USER_APPLICATIONS)}: {query}")
            
            repos = self.search_repositories(query, max_results=100)
            self.search_stats["total_found"] += len(repos)
            
            if not repos:
                continue
            
            logger.info(f"Found {len(repos)} repos, strict filtering...")
            
            for repo in tqdm(repos, desc=f"Filtering {i+1}/{len(AGENTIC_USER_APPLICATIONS)}"):
                repo_id = repo['id']
                
                if repo_id in self.discovered_repos:
                    self.search_stats["duplicates_removed"] += 1
                    continue
                
                # Quality check first
                if not self.check_quality_criteria(repo):
                    self.search_stats["quality_failed"] += 1
                    continue
                
                self.search_stats["quality_passed"] += 1
                
                # Get README for detailed analysis
                readme_content = self.get_repository_readme(repo['full_name'])
                repo['readme_content'] = readme_content
                
                # Strict application filtering
                is_app, reason, confidence = self.is_user_facing_application(repo)
                
                if not is_app:
                    if "RAG-focused" in reason:
                        self.search_stats["rag_excluded"] += 1
                    else:
                        self.search_stats["frameworks_excluded"] += 1
                    continue
                
                # Classify type
                chatbot_type, type_confidence = self.classify_chatbot_type(repo)
                
                # Create processed repo
                processed_repo = {
                    **repo,
                    'chatbot_type': chatbot_type,
                    'type_confidence': type_confidence,
                    'application_confidence': confidence,
                    'application_reason': reason,
                    'overall_confidence': (confidence + type_confidence) / 2
                }
                
                all_applications.append(processed_repo)
                self.discovered_repos[repo_id] = processed_repo
                self.search_stats["applications_identified"] += 1
            
            logger.info(f"Query {i+1}: {self.search_stats['applications_identified']} total apps")
            time.sleep(2)
        
        # Group by type
        chatbot_types = {"General": [], "Agent": []}
        
        for app in all_applications:
            chatbot_type = app['chatbot_type']
            chatbot_types[chatbot_type].append(app)
        
        # Sort by confidence
        for chatbot_type in chatbot_types:
            chatbot_types[chatbot_type].sort(
                key=lambda x: (x['overall_confidence'], x['stargazers_count']),
                reverse=True
            )
        
        return chatbot_types

    def save_results(self, chatbot_types: Dict[str, List[Dict]], output_dir: str = "output"):
        """Save results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        all_apps = []
        for chatbot_type, repos in chatbot_types.items():
            all_apps.extend(repos)
        
        if not all_apps:
            logger.warning("No user-facing chatbot applications found")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_apps)
        
        columns = [
            'id', 'name', 'full_name', 'description', 'html_url',
            'stargazers_count', 'contributor_count', 'commit_count_estimate',
            'chatbot_type', 'type_confidence', 'application_confidence', 
            'overall_confidence', 'application_reason'
        ]
        
        available_columns = [col for col in columns if col in df.columns]
        df_clean = df[available_columns]
        df_clean = df_clean.sort_values(['chatbot_type', 'overall_confidence'], ascending=[True, False])
        
        df_clean['selected'] = ''
        df_clean['final_notes'] = ''
        
        df_clean.to_csv(f"{output_dir}/user_facing_chatbots.csv", index=False)
        
        # Selection candidates
        selection_data = []
        for chatbot_type, repos in chatbot_types.items():
            for rank, repo in enumerate(repos[:40], 1):  # Top 40 per type
                selection_data.append({
                    'rank': rank,
                    'type': chatbot_type,
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'description': repo.get('description', ''),
                    'stars': repo['stargazers_count'],
                    'confidence': round(repo['overall_confidence'], 3),
                    'reason': repo['application_reason'],
                    'html_url': repo['html_url'],
                    'selected': ''
                })
        
        df_selection = pd.DataFrame(selection_data)
        df_selection.to_csv(f"{output_dir}/final_chatbot_candidates.csv", index=False)
        
        # Print results
        print("\n" + "="*70)
        print("USER-FACING CHATBOT DISCOVERY RESULTS")
        print("="*70)
        print("STRICT FILTERING: Only end-user applications (NO frameworks/RAG)")
        print(f"Total user applications: {len(all_apps)}")
        print(f"Frameworks excluded: {self.search_stats['frameworks_excluded']}")
        print(f"RAG apps excluded: {self.search_stats['rag_excluded']}")
        print("-"*70)
        
        for chatbot_type, repos in chatbot_types.items():
            high_conf = len([r for r in repos if r['overall_confidence'] >= 0.6])
            print(f"{chatbot_type:8}: {len(repos):3d} total, {high_conf:3d} high-confidence")
        
        print("="*70)
        print("Output files:")
        print(f"  📊 user_facing_chatbots.csv - All {len(all_apps)} user applications")
        print(f"  🎯 final_chatbot_candidates.csv - Top 40 per type")
        print("="*70)
        print("SELECTION GUIDE:")
        print("1. Review final_chatbot_candidates.csv")
        print("2. Select 30 repositories per type:")
        print("   - General: Basic chat UIs, text generators, conversation apps")
        print("   - Agent: Personal assistants, task automation, autonomous bots")
        print("3. Mark 'selected' = 'Yes'")
        print("4. Target: 60 user-facing chatbot applications")
        print("="*70)

def main():
    """Main execution"""
    if not GITHUB_TOKEN:
        print("Error: Please set GITHUB_TOKEN environment variable")
        return
    
    print("🎯 User-Facing Chatbot Discovery")
    print("Target: General + Agent chatbots ONLY (no frameworks, no RAG)")
    print("Focus: True end-user applications for public use")
    print()
    
    criteria = QualityCriteria()
    print("Very Relaxed Quality Criteria:")
    print(f"  Stars: >={criteria.min_stars}")
    print(f"  Age: >={criteria.min_age_days} days")
    print(f"  Contributors: >{criteria.min_contributors}")
    print()
    
    discovery = UserFacingChatbotDiscovery(GITHUB_TOKEN, criteria)
    
    try:
        chatbot_types = discovery.discover_user_chatbots()
        discovery.save_results(chatbot_types)
        print("\n✅ User-facing chatbot discovery completed!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted")
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise

if __name__ == "__main__":
    main()
