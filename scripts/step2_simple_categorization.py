#!/usr/bin/env python3
"""
Step 2: Optimal 3-Signal Repository Categorization
Balance academic rigor with practical efficiency

Requirements:
pip install pandas numpy requests python-dotenv ratelimit
"""

import json
import pandas as pd
import numpy as np
import requests
import base64
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from ratelimit import limits, sleep_and_retry
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Category Keywords with confidence weights
CATEGORY_KEYWORDS = {
    "1A_2A": {  # Training Infrastructure
        "strong": ["lora", "qlora", "peft", "fine-tuning", "rlhf", "dpo", "adapter"],
        "medium": ["training", "instruction-tuning", "alignment", "parameter-efficient"],
        "weak": ["deepspeed", "accelerate", "torch", "transformers"]
    },
    "1A_2B": {  # Serving Infrastructure  
        "strong": ["vllm", "text-generation-inference", "tgi", "inference-server"],
        "medium": ["quantization", "model-serving", "serving", "deployment"],
        "weak": ["fastapi", "uvicorn", "optimization", "gpu"]
    },
    "1A_2C": {  # RAG Infrastructure
        "strong": ["langchain", "llamaindex", "rag", "vector-database", "retrieval"],
        "medium": ["haystack", "semantic-kernel", "embedding", "faiss", "chroma"],
        "weak": ["pinecone", "document", "knowledge", "search"]
    },
    "1A_2D": {  # Agentic Infrastructure
        "strong": ["autogen", "crewai", "langraph", "multi-agent", "agent-framework"],
        "medium": ["workflow", "orchestration", "tool-calling", "planning"],
        "weak": ["autonomous", "coordination", "task", "reasoning"]
    },
    "1A_2E": {  # Evaluation Infrastructure
        "strong": ["lm-eval", "eval-harness", "benchmark", "evaluation-framework"],
        "medium": ["testing", "metrics", "evaluation", "assessment"],
        "weak": ["quality", "performance", "analysis"]
    },
    "1B_2B": {  # Serving Applications
        "strong": ["chatbot-ui", "chat-interface", "chat-app", "playground"],
        "medium": ["text-generator", "conversation", "chatbot", "ui"],
        "weak": ["frontend", "streamlit", "gradio", "interface"]
    },
    "1B_2C": {  # RAG Applications
        "strong": ["chat-with-pdf", "document-chat", "ask-pdf", "pdf-chat"],
        "medium": ["document-assistant", "knowledge-chat", "search-app"],
        "weak": ["wiki", "notion", "document", "knowledge"]
    },
    "1B_2D": {  # Agentic Applications
        "strong": ["auto-gpt", "personal-assistant", "ai-assistant", "baby-agi"],
        "medium": ["task-automation", "autonomous-agent", "assistant"],
        "weak": ["productivity", "workflow", "automation", "agent"]
    }
}

# Key dependencies that strongly indicate category
DEPENDENCY_INDICATORS = {
    "1A_2A": ["torch", "transformers", "peft", "accelerate", "deepspeed", "wandb", "trl"],
    "1A_2B": ["vllm", "torch", "transformers", "fastapi", "uvicorn", "nvidia-ml-py"],
    "1A_2C": ["langchain", "llamaindex", "faiss-cpu", "chromadb", "pinecone-client", "sentence-transformers"],
    "1A_2D": ["autogen-agentchat", "crewai", "langchain", "openai", "anthropic"],
    "1A_2E": ["datasets", "evaluate", "rouge-score", "bleu", "sacrebleu", "pytest"],
    "1B_2B": ["streamlit", "gradio", "fastapi", "flask", "chainlit", "nicegui", "dash"],
    "1B_2C": ["langchain", "llamaindex", "streamlit", "pypdf", "unstructured", "gradio"],
    "1B_2D": ["langchain", "openai", "anthropic", "streamlit", "fastapi", "requests"]
}

class OptimalCategorizer:
    def __init__(self, json_file_path: str, github_token: str):
        self.json_file_path = json_file_path
        self.token = github_token
        self.headers = {"Authorization": f"token {github_token}"}
        self.base_url = "https://api.github.com"
        self.repos = []

    def load_repositories(self) -> List[Dict]:
        """Load repositories from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.repos = json.load(f)
            logger.info(f"Loaded {len(self.repos)} repositories for categorization")
            return self.repos
        except Exception as e:
            logger.error(f"Error loading repositories: {e}")
            return []

    @sleep_and_retry
    @limits(calls=5000, period=3600)
    def get_dependencies(self, full_name: str) -> List[str]:
        """Get repository dependencies from requirements.txt or setup.py"""
        requirements_files = ['requirements.txt', 'setup.py', 'pyproject.toml']
        
        for req_file in requirements_files:
            try:
                url = f"{self.base_url}/repos/{full_name}/contents/{req_file}"
                response = requests.get(url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    file_data = response.json()
                    content = base64.b64decode(file_data['content']).decode('utf-8')
                    
                    # Extract package names (simple parsing)
                    dependencies = []
                    for line in content.split('\n'):
                        line = line.strip().lower()
                        if line and not line.startswith('#'):
                            # Extract package name before version specifiers
                            pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                            pkg_name = pkg_name.split('[')[0].strip()  # Remove extras like [gpu]
                            if pkg_name:
                                dependencies.append(pkg_name)
                    
                    return dependencies
            except Exception:
                continue
        
        return []

    def analyze_metadata_signal(self, repo: Dict) -> Tuple[str, str, float]:
        """Analyze metadata: name, description, topics"""
        name = repo.get('name', '').lower()
        description = repo.get('description', '').lower()
        topics = ' '.join(repo.get('topics', [])).lower()
        
        metadata_text = f"{name} {description} {topics}"
        
        # Score each category
        category_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = 0
            
            # Strong indicators (weight: 3)
            for keyword in keywords["strong"]:
                score += metadata_text.count(keyword) * 3
            
            # Medium indicators (weight: 2)
            for keyword in keywords["medium"]:
                score += metadata_text.count(keyword) * 2
                
            # Weak indicators (weight: 1)
            for keyword in keywords["weak"]:
                score += metadata_text.count(keyword) * 1
            
            category_scores[category] = score
        
        # Get best category
        best_category = max(category_scores, key=category_scores.get) if category_scores else "1A_2B"
        best_score = category_scores.get(best_category, 0)
        
        layer_1, layer_2 = best_category.split('_')
        confidence = min(best_score / 10.0, 1.0)  # Normalize
        
        return layer_1, layer_2, confidence

    def analyze_dependencies_signal(self, repo: Dict) -> Tuple[str, str, float]:
        """Analyze dependencies for strong classification signal"""
        try:
            dependencies = self.get_dependencies(repo['full_name'])
            
            if not dependencies:
                return "1A", "2B", 0.0
            
            deps_text = ' '.join(dependencies).lower()
            
            # Score categories based on dependency patterns
            category_scores = {}
            for category, dep_indicators in DEPENDENCY_INDICATORS.items():
                score = sum(1 for dep in dep_indicators if dep in deps_text)
                category_scores[category] = score
            
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                best_score = category_scores[best_category]
                
                if best_score > 0:
                    layer_1, layer_2 = best_category.split('_')
                    # High confidence for dependency matches
                    confidence = min(best_score / len(DEPENDENCY_INDICATORS[best_category]), 1.0)
                    return layer_1, layer_2, confidence
            
            return "1A", "2B", 0.0
            
        except Exception as e:
            logger.debug(f"Dependencies analysis failed for {repo['full_name']}: {e}")
            return "1A", "2B", 0.0

    def analyze_readme_signal(self, repo: Dict) -> Tuple[str, str, float]:
        """Analyze README content for classification"""
        readme_content = repo.get('readme_content', '').lower()
        
        if not readme_content or len(readme_content) < 100:
            return "1A", "2B", 0.0
        
        # Score each category with keyword frequency
        category_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = 0
            
            # Count occurrences with weights
            for keyword in keywords["strong"]:
                score += readme_content.count(keyword.lower()) * 3
            for keyword in keywords["medium"]:
                score += readme_content.count(keyword.lower()) * 2
            for keyword in keywords["weak"]:
                score += readme_content.count(keyword.lower()) * 1
            
            category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            
            if best_score > 0:
                layer_1, layer_2 = best_category.split('_')
                # Normalize by README length
                confidence = min(best_score / max(len(readme_content.split()) * 0.05, 1), 1.0)
                return layer_1, layer_2, confidence
        
        return "1A", "2B", 0.0

    def classify_infrastructure_vs_application(self, repo: Dict) -> str:
        """Determine Layer 1: Infrastructure (1A) vs Application (1B)"""
        name = repo.get('name', '').lower()
        description = repo.get('description', '').lower()
        readme = repo.get('readme_content', '').lower()
        
        # Strong infrastructure indicators
        infra_indicators = [
            "framework", "library", "toolkit", "sdk", "engine", "api", "client",
            "core", "base", "driver", "wrapper", "connector", "package"
        ]
        
        # Strong application indicators
        app_indicators = [
            "ui", "interface", "app", "chatbot", "assistant", "chat", "frontend",
            "webapp", "dashboard", "studio", "playground", "demo", "tool"
        ]
        
        # Check in name (strongest signal)
        name_infra = sum(1 for ind in infra_indicators if ind in name)
        name_app = sum(1 for ind in app_indicators if ind in name)
        
        if name_infra > name_app:
            return "1A"
        elif name_app > name_infra:
            return "1B"
        
        # Check in description
        desc_infra = sum(1 for ind in infra_indicators if ind in description)
        desc_app = sum(1 for ind in app_indicators if ind in description)
        
        if desc_infra > desc_app:
            return "1A"
        elif desc_app > desc_infra:
            return "1B"
        
        # Check in README (last resort)
        readme_infra = sum(1 for ind in infra_indicators if ind in readme)
        readme_app = sum(1 for ind in app_indicators if ind in readme)
        
        return "1A" if readme_infra >= readme_app else "1B"

    def three_signal_classification(self, repo: Dict) -> Dict:
        """Classify using optimal 3-signal approach"""
        
        # Signal 1: Metadata + Topics (weight: 1.0)
        meta_l1, meta_l2, meta_conf = self.analyze_metadata_signal(repo)
        
        # Signal 2: Dependencies (weight: 2.5 - highest weight)
        dep_l1, dep_l2, dep_conf = self.analyze_dependencies_signal(repo)
        
        # Signal 3: README (weight: 1.5)
        readme_l1, readme_l2, readme_conf = self.analyze_readme_signal(repo)
        
        # Override Layer 1 with infrastructure vs application analysis
        final_layer_1 = self.classify_infrastructure_vs_application(repo)
        
        # Weighted voting for Layer 2
        layer_2_scores = {}
        votes = [
            (meta_l2, meta_conf * 1.0),
            (dep_l2, dep_conf * 2.5),    # Dependencies most reliable
            (readme_l2, readme_conf * 1.5)
        ]
        
        for l2, weight in votes:
            if weight > 0:
                layer_2_scores[l2] = layer_2_scores.get(l2, 0) + weight
        
        final_layer_2 = max(layer_2_scores, key=layer_2_scores.get) if layer_2_scores else "2B"
        
        # Calculate overall confidence
        total_weight = meta_conf * 1.0 + dep_conf * 2.5 + readme_conf * 1.5
        max_weight = 5.0  # 1.0 + 2.5 + 1.5
        final_confidence = min(total_weight / max_weight, 1.0)
        
        final_category = f"{final_layer_1}_{final_layer_2}"
        
        return {
            **repo,
            'layer_1': final_layer_1,
            'layer_2': final_layer_2,
            'category': final_category,
            'classification_confidence': final_confidence,
            'metadata_confidence': meta_conf,
            'dependency_confidence': dep_conf,
            'readme_confidence': readme_conf,
            'classification_method': '3_signal_optimal',
            'signal_agreement': self.calculate_signal_agreement(
                [(meta_l1, meta_l2), (dep_l1, dep_l2), (readme_l1, readme_l2)]
            )
        }

    def calculate_signal_agreement(self, classifications: List[Tuple[str, str]]) -> float:
        """Calculate how much signals agree (for academic validation)"""
        if not classifications:
            return 0.0
        
        # Check Layer 1 agreement
        layer_1_votes = [l1 for l1, l2 in classifications]
        layer_1_agreement = layer_1_votes.count(max(set(layer_1_votes), key=layer_1_votes.count)) / len(layer_1_votes)
        
        # Check Layer 2 agreement  
        layer_2_votes = [l2 for l1, l2 in classifications]
        layer_2_agreement = layer_2_votes.count(max(set(layer_2_votes), key=layer_2_votes.count)) / len(layer_2_votes)
        
        return (layer_1_agreement + layer_2_agreement) / 2.0

    def categorize_all_repositories(self) -> Dict[str, List[Dict]]:
        """Categorize all repositories using 3-signal approach"""
        logger.info("Starting 3-signal categorization for all repositories...")
        logger.info("Signals: (1) Metadata+Topics, (2) Dependencies, (3) README")
        
        categorized_by_category = {cat: [] for cat in CATEGORY_KEYWORDS.keys()}
        categorized_repos = []
        
        for i, repo in enumerate(self.repos):
            if i % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(self.repos)} repositories")
            
            try:
                categorized_repo = self.three_signal_classification(repo)
                category = categorized_repo['category']
                
                if category in categorized_by_category:
                    categorized_by_category[category].append(categorized_repo)
                
                categorized_repos.append(categorized_repo)
                
            except Exception as e:
                logger.error(f"Failed to categorize {repo['full_name']}: {e}")
                continue
        
        # Sort each category by confidence and signal agreement
        for category in categorized_by_category:
            categorized_by_category[category].sort(
                key=lambda x: (x['classification_confidence'], x['signal_agreement'], x['stargazers_count']), 
                reverse=True
            )
        
        self.categorized_repos = categorized_repos
        return categorized_by_category

    def save_results(self, categorized_by_category: Dict[str, List[Dict]], output_dir: str = "output"):
        """Save categorization results with academic transparency"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Complete analysis CSV
        df_all = pd.DataFrame(self.categorized_repos)
        df_all = df_all.sort_values(['category', 'classification_confidence', 'signal_agreement'], 
                                   ascending=[True, False, False])
        
        csv_columns = [
            'id', 'name', 'full_name', 'description', 'html_url', 'stargazers_count',
            'layer_1', 'layer_2', 'category', 'classification_confidence', 'signal_agreement',
            'metadata_confidence', 'dependency_confidence', 'readme_confidence',
            'classification_method', 'quality_score'
        ]
        
        available_columns = [col for col in csv_columns if col in df_all.columns]
        df_csv = df_all[available_columns]
        df_csv.to_csv(f"{output_dir}/3_signal_categorized_repositories.csv", index=False)
        
        # Selection-ready CSV (top candidates per category)
        selection_data = []
        for category, repos in categorized_by_category.items():
            for rank, repo in enumerate(repos[:40], 1):  # Top 40 per category
                selection_data.append({
                    'rank': rank,
                    'category': category,
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'description': repo.get('description', ''),
                    'stars': repo['stargazers_count'],
                    'confidence': round(repo['classification_confidence'], 3),
                    'signal_agreement': round(repo['signal_agreement'], 3),
                    'metadata_conf': round(repo['metadata_confidence'], 3),
                    'dependency_conf': round(repo['dependency_confidence'], 3),
                    'readme_conf': round(repo['readme_confidence'], 3),
                    'html_url': repo['html_url'],
                    'selected': '',  # For manual selection
                    'notes': ''
                })
        
        df_selection = pd.DataFrame(selection_data)
        df_selection.to_csv(f"{output_dir}/ranked_selection_candidates.csv", index=False)
        
        # Academic validation data
        validation_stats = self.calculate_validation_statistics(categorized_by_category)
        with open(f"{output_dir}/classification_validation_stats.json", 'w') as f:
            json.dump(validation_stats, f, indent=2)
        
        # Print academic summary
        print("\n" + "="*80)
        print("3-SIGNAL REPOSITORY CATEGORIZATION SUMMARY")
        print("="*80)
        print("Classification methodology:")
        print("  Signal 1: Metadata + Topics (weight: 1.0)")
        print("  Signal 2: Dependencies analysis (weight: 2.5) 🔑 Most reliable")
        print("  Signal 3: README content analysis (weight: 1.5)")
        print("  Layer 1 override: Infrastructure vs Application heuristics")
        print("-"*80)
        print("Results by category (confidence ≥ 0.5):")
        
        total_high_conf = 0
        for category, repos in categorized_by_category.items():
            high_conf = [r for r in repos if r['classification_confidence'] >= 0.5]
            very_high_conf = [r for r in repos if r['classification_confidence'] >= 0.7]
            
            print(f"  {category}: {len(repos):3d} total, {len(high_conf):3d} conf≥0.5, {len(very_high_conf):3d} conf≥0.7")
            total_high_conf += len(high_conf)
        
        print("-"*80)
        print(f"Total repositories: {len(self.categorized_repos)}")
        print(f"High confidence (≥0.5): {total_high_conf}")
        print(f"Coverage: {total_high_conf/len(self.categorized_repos)*100:.1f}%")
        print("="*80)
        print("\n📁 Academic-grade outputs:")
        print("  📊 3_signal_categorized_repositories.csv - Complete analysis")
        print("  🎯 ranked_selection_candidates.csv - Top 40 per category") 
        print("  📈 classification_validation_stats.json - Academic validation metrics")
        print("="*80)
        print("\n✅ ACADEMIC BENEFITS:")
        print("  🔬 Multi-signal triangulation (not just README)")
        print("  📊 Confidence scoring for quality control")
        print("  🤝 Signal agreement measurement for validation")
        print("  📝 Transparent methodology for peer review")
        print("  🎯 Dependency analysis = strongest technical signal")
        print("="*80)

    def calculate_validation_statistics(self, categorized_by_category: Dict) -> Dict:
        """Calculate statistics for academic validation"""
        stats = {
            "total_repositories": len(self.categorized_repos),
            "categories": {}
        }
        
        for category, repos in categorized_by_category.items():
            category_stats = {
                "total_count": len(repos),
                "high_confidence_count": len([r for r in repos if r['classification_confidence'] >= 0.7]),
                "medium_confidence_count": len([r for r in repos if 0.5 <= r['classification_confidence'] < 0.7]),
                "low_confidence_count": len([r for r in repos if r['classification_confidence'] < 0.5]),
                "avg_confidence": np.mean([r['classification_confidence'] for r in repos]) if repos else 0,
                "avg_signal_agreement": np.mean([r['signal_agreement'] for r in repos]) if repos else 0
            }
            stats["categories"][category] = category_stats
        
        return stats

def main():
    """Main execution function"""
    json_file = "output_step1/repositories_with_readme.json"
    
    if not Path(json_file).exists():
        print(f"❌ Error: {json_file} not found")
        print("Please run step1_simple_discovery.py first")
        return
    
    if not GITHUB_TOKEN:
        print("❌ Error: GitHub token required for dependency analysis")
        return
    
    print("🧠 Optimal 3-Signal Repository Categorization")
    print(f"📁 Input: {json_file}")
    print("🎯 Goal: Academically rigorous classification with practical efficiency")
    print("⏱️  Estimated time: 15-25 minutes for 874 repositories")
    print()
    print("🔬 Classification signals:")
    print("  1. Metadata + Topics (instant)")
    print("  2. Dependencies analysis (most reliable - requires API)")
    print("  3. README content (instant)")
    print()
    
    # Initialize categorizer
    categorizer = OptimalCategorizer(json_file, GITHUB_TOKEN)
    
    try:
        # Load repositories
        repos = categorizer.load_repositories()
        if not repos:
            print("❌ No repositories loaded")
            return
        
        # 3-signal categorization
        categorized_by_category = categorizer.categorize_all_repositories()
        
        # Save results
        categorizer.save_results(categorized_by_category)
        
        print("\n✅ 3-signal categorization completed!")
        print("📊 Ready for repository selection!")
        
    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        print(f"❌ Categorization failed: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Step 2: Automatic Repository Categorization
Use keywords to automatically categorize LLM repositories from Step 1

Requirements:
pip install pandas numpy
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Category Keywords (from our refined discussion)
CATEGORY_KEYWORDS = {
    "1A_2A": [  # Training Infrastructure
        "fine-tuning", "finetune", "finetuning", "lora", "qlora", "peft", 
        "parameter-efficient", "adapter-tuning", "prompt-tuning", "prefix-tuning",
        "instruction-tuning", "supervised-finetuning", "sft", "rlhf", "dpo", 
        "alignment", "preference-tuning", "reward-model", "deepspeed", "accelerate",
        "training-framework", "finetuning-toolkit", "model-training"
    ],
    
    "1A_2B": [  # Serving Infrastructure
        "vllm", "text-generation-inference", "tgi", "triton-inference-server",
        "tensorrt-llm", "quantization", "inference-server", "model-serving",
        "serving-framework", "inference-engine", "tensor-parallelism",
        "continuous-batching", "dynamic-batching", "model-deployment"
    ],
    
    "1A_2C": [  # RAG Infrastructure
        "rag-framework", "retrieval-framework", "llamaindex", "langchain",
        "haystack", "semantic-kernel", "vector-database", "vector-store",
        "embedding-store", "faiss", "pinecone", "chroma", "qdrant", "weaviate",
        "document-loader", "text-splitter", "semantic-search", "hybrid-search"
    ],
    
    "1A_2D": [  # Agentic Infrastructure
        "autogen", "crewai", "langraph", "multi-agent", "agent-framework",
        "workflow-engine", "agent-orchestration", "tool-calling", "function-calling",
        "task-planning", "agent-coordination", "planning-framework"
    ],
    
    "1A_2E": [  # Evaluation Infrastructure
        "eval-harness", "lm-eval", "evaluation-framework", "benchmark-suite",
        "llm-testing", "model-testing", "evaluation-metrics", "hallucination-detection"
    ],
    
    "1B_2B": [  # Serving Applications
        "chatbot-ui", "chat-interface", "chat-app", "text-generator",
        "prompt-playground", "llm-playground", "conversation-ui", "ai-writer",
        "streamlit-chat", "gradio-chat", "simple-chat", "basic-chatbot"
    ],
    
    "1B_2C": [  # RAG Applications
        "chat-with-pdf", "document-chat", "pdf-chat", "ask-pdf",
        "document-assistant", "knowledge-chat", "semantic-search-app",
        "ai-search", "wiki-chat", "notion-chat"
    ],
    
    "1B_2D": [  # Agentic Applications
        "ai-assistant", "personal-assistant", "auto-gpt", "baby-agi",
        "autonomous-agent", "task-automation", "code-assistant", 
        "research-assistant", "productivity-agent", "workflow-automation"
    ]
}

# Layer 1 indicators
INFRASTRUCTURE_INDICATORS = [
    "framework", "library", "toolkit", "sdk", "engine", "server-engine",
    "inference-server", "training-framework", "core-library", "base-engine",
    "api", "client", "wrapper", "driver", "connector"
]

APPLICATION_INDICATORS = [
    "chat-app", "web-ui", "user-interface", "bot-app", "dashboard-app",
    "studio-app", "playground-app", "chatbot", "assistant", "ui", "frontend",
    "webapp", "application", "tool", "demo"
]

class RepositoryCategorizer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.repos = []
        self.categorized_repos = []
        
    def load_repositories(self) -> List[Dict]:
        """Load repositories from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.repos = json.load(f)
            logger.info(f"Loaded {len(self.repos)} repositories")
            return self.repos
        except Exception as e:
            logger.error(f"Error loading repositories: {e}")
            return []
    
    def classify_layer_1(self, repo: Dict) -> str:
        """Classify Layer 1: Infrastructure (1A) vs Application (1B)"""
        text = repo.get('combined_text', '').lower()
        
        # Count infrastructure indicators
        infra_score = sum(1 for indicator in INFRASTRUCTURE_INDICATORS if indicator in text)
        
        # Count application indicators  
        app_score = sum(1 for indicator in APPLICATION_INDICATORS if indicator in text)
        
        # Special case: if repo name contains clear indicators
        name = repo.get('name', '').lower()
        if any(word in name for word in ['ui', 'chat', 'bot', 'app']):
            app_score += 2
        if any(word in name for word in ['framework', 'library', 'sdk', 'engine']):
            infra_score += 2
        
        # Decision
        if infra_score > app_score:
            return "1A"
        elif app_score > infra_score:
            return "1B"
        else:
            # Tie-breaker: check description more carefully
            description = repo.get('description', '').lower()
            if any(word in description for word in ['framework', 'library', 'tool for developers']):
                return "1A"
            else:
                return "1B"  # Default to application if unclear
    
    def classify_layer_2(self, repo: Dict) -> str:
        """Classify Layer 2: Function (2A, 2B, 2C, 2D, 2E)"""
        text = repo.get('combined_text', '').lower()
        
        # Score each function category
        function_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            layer_2 = category.split('_')[1]  # Extract 2A, 2B, etc.
            
            if layer_2 not in function_scores:
                function_scores[layer_2] = 0
            
            # Count keyword matches
            for keyword in keywords:
                if keyword.lower() in text:
                    function_scores[layer_2] += 1
        
        # Return function with highest score
        if function_scores:
            best_function = max(function_scores, key=function_scores.get)
            return best_function
        else:
            return "2B"  # Default to serving if no clear matches
    
    def calculate_confidence_score(self, repo: Dict, category: str) -> float:
        """Calculate confidence score for the classification"""
        text = repo.get('combined_text', '').lower()
        
        if category not in CATEGORY_KEYWORDS:
            return 0.0
        
        keywords = CATEGORY_KEYWORDS[category]
        matches = sum(1 for keyword in keywords if keyword.lower() in text)
        
        # Normalize by keyword count
        confidence = matches / len(keywords)
        
        # Boost confidence for strong indicators
        name = repo.get('name', '').lower()
        description = repo.get('description', '').lower()
        
        strong_indicators = {
            "1A_2A": ["lora", "peft", "fine-tuning"],
            "1A_2B": ["vllm", "inference", "serving"],
            "1A_2C": ["langchain", "llamaindex", "rag"],
            "1A_2D": ["autogen", "crewai", "agent"],
            "1A_2E": ["eval", "benchmark", "testing"],
            "1B_2B": ["chatbot", "chat-ui", "playground"],
            "1B_2C": ["chat-pdf", "document-chat", "ask-pdf"],
            "1B_2D": ["auto-gpt", "assistant", "automation"]
        }
        
        if category in strong_indicators:
            for indicator in strong_indicators[category]:
                if indicator in name or indicator in description:
                    confidence += 0.3
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def categorize_repository(self, repo: Dict) -> Dict:
        """Categorize single repository"""
        # Classify layers
        layer_1 = self.classify_layer_1(repo)
        layer_2 = self.classify_layer_2(repo)
        category = f"{layer_1}_{layer_2}"
        
        # Calculate confidence
        confidence = self.calculate_confidence_score(repo, category)
        
        # Create categorized repo
        categorized = {
            **repo,
            'layer_1': layer_1,
            'layer_2': layer_2, 
            'category': category,
            'confidence_score': confidence,
            'auto_classified': True
        }
        
        return categorized
    
    def categorize_all_repositories(self) -> Dict[str, List[Dict]]:
        """Categorize all repositories and group by category"""
        logger.info("Starting automatic categorization...")
        
        categorized_by_category = {cat: [] for cat in CATEGORY_KEYWORDS.keys()}
        
        for repo in self.repos:
            categorized_repo = self.categorize_repository(repo)
            category = categorized_repo['category']
            
            if category in categorized_by_category:
                categorized_by_category[category].append(categorized_repo)
            
            self.categorized_repos.append(categorized_repo)
        
        # Sort each category by confidence score
        for category in categorized_by_category:
            categorized_by_category[category].sort(
                key=lambda x: (x['confidence_score'], x['stargazers_count']), 
                reverse=True
            )
        
        return categorized_by_category
    
    def save_categorized_results(self, categorized_by_category: Dict[str, List[Dict]], 
                               output_dir: str = "output"):
        """Save categorization results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save complete categorized data
        df_all = pd.DataFrame(self.categorized_repos)
        df_all = df_all.sort_values(['category', 'confidence_score', 'stargazers_count'], 
                                   ascending=[True, False, False])
        
        # Select relevant columns for CSV
        csv_columns = [
            'id', 'name', 'full_name', 'description', 'html_url',
            'stargazers_count', 'contributor_count', 'commit_count_estimate',
            'layer_1', 'layer_2', 'category', 'confidence_score',
            'discovery_query', 'quality_score'
        ]
        
        available_columns = [col for col in csv_columns if col in df_all.columns]
        df_csv = df_all[available_columns]
        
        # Add selection columns
        df_csv['selected'] = ''
        df_csv['notes'] = ''
        
        df_csv.to_csv(f"{output_dir}/auto_categorized_repositories.csv", index=False)
        
        # Save top candidates per category for manual selection
        selection_data = []
        for category, repos in categorized_by_category.items():
            top_repos = repos[:30]  # Top 30 candidates per category
            for repo in top_repos:
                selection_data.append({
                    'category': category,
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'description': repo.get('description', ''),
                    'stars': repo['stargazers_count'],
                    'confidence': repo['confidence_score'],
                    'html_url': repo['html_url'],
                    'selected': '',  # For manual selection
                    'notes': ''
                })
        
        df_selection = pd.DataFrame(selection_data)
        df_selection.to_csv(f"{output_dir}/top_candidates_for_selection.csv", index=False)
        
        # Save complete JSON for potential re-analysis
        with open(f"{output_dir}/categorized_repositories.json", 'w', encoding='utf-8') as f:
            json.dump(categorized_by_category, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*70)
        print("AUTOMATIC CATEGORIZATION SUMMARY")
        print("="*70)
        print("Repositories per category:")
        total_categorized = 0
        for category, repos in categorized_by_category.items():
            print(f"  {category}: {len(repos):3d} repositories")
            total_categorized += len(repos)
        print("-"*70)
        print(f"Total categorized: {total_categorized}")
        print("="*70)
        print("\n📁 Output files:")
        print("  📊 auto_categorized_repositories.csv - All categorized repos")
        print("  🎯 top_candidates_for_selection.csv - Top 30 per category for selection")
        print("  🔍 categorized_repositories.json - Complete data by category")
        print("="*70)
        print("\n📋 MANUAL SELECTION STEPS:")
        print("1. Open top_candidates_for_selection.csv")
        print("2. For each category, select 20 best repositories:")
        print("   - Mark 'selected' column as 'Yes' for chosen repos")
        print("   - Consider: confidence score, stars, description quality")
        print("   - Aim for exactly 20 per category")
        print("3. Save as final_selected_repositories.csv")
        print("4. Proceed to Step 3: Issues Data Collection")
        print("="*70)

def main():
    """Main execution function"""
    json_file = "output_step1/repositories_with_readme.json"
    
    if not Path(json_file).exists():
        print(f"❌ Error: {json_file} not found")
        print("Please run step1_simple_discovery.py first")
        return
    
    print("🤖 Automatic Repository Categorization - Step 2")
    print(f"📁 Input: {json_file}")
    print("🎯 Goal: Auto-categorize repositories using keywords")
    print()
    
    # Initialize categorizer
    categorizer = RepositoryCategorizer(json_file)
    
    try:
        # Load repositories
        repos = categorizer.load_repositories()
        if not repos:
            print("❌ No repositories loaded")
            return
        
        # Categorize repositories
        categorized_by_category = categorizer.categorize_all_repositories()
        
        # Save results
        categorizer.save_categorized_results(categorized_by_category)
        
        print("\n✅ Automatic categorization completed!")
        print("📝 Review top_candidates_for_selection.csv for manual selection")
        
    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        print(f"❌ Categorization failed: {e}")
        raise

if __name__ == "__main__":
    main()
