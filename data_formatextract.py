import json
import pandas as pd
from typing import List, Dict

def create_complete_repository_input():
    """Convert your complete manual validation data (165 repos) into extraction format - ARCHITECTURE REMOVED"""
    
    repositories = [
        # 1A + 2A: Training & Fine-tuning Infrastructure (18 repos)
        {"url": "https://github.com/adapter-hub/adapters", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/OpenRLHF/OpenRLHF", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/PKU-Alignment/safe-rlhf", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/oumi-ai/oumi", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/openaccess-ai-collective/axolotl", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/sousa-lab/GALOR", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/huggingface/trl", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/unslothai/unsloth", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/horseee/LLM-Pruner", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/lucidrains/self-rewarding-lm-pytorch", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/mlabonne/dpo-from-scratch", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/AGIS-Lab/LLM-Adapters", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/hiyouga/LLaMA-Factory", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/generative-ai-utah/long-context", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/pytorch/torchtune", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/allenai/lm-buddy", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/mosaicml/llm-foundry", "scope": "Infrastructure/Tool", "function": "Training"},
        {"url": "https://github.com/InternLM/xtuner", "scope": "Infrastructure/Tool", "function": "Training"},
        
        # 1A + 2B: Serving & Inference Infrastructure (16 repos)
        {"url": "https://github.com/gpustack/gpustack", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/NVIDIA/TensorRT-Model-Optimizer", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/ModelTC/LightLLM", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/ModelCloud/GPTQModel", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/SeldonIO/MLServer", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/bentoml/OpenLLM", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/microsoft/sarathi-serve", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/NVIDIA/NeMo-Guardrails", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/intel/auto-round", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/vllm-project/vllm", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/zenml-io/zenml", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/PaddlePaddle/FastDeploy", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/FunAudioLLM/CosyVoice", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/Emerging-AI/ENOVA", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/allwefantasy/byzer-llm", "scope": "Infrastructure/Tool", "function": "Serving"},
        {"url": "https://github.com/intel/neural-compressor", "scope": "Infrastructure/Tool", "function": "Serving"},
        
        # 1A + 2C: RAG & Knowledge Infrastructure (20 repos)
        {"url": "https://github.com/IntelLabs/RAG-FiT", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/castorini/pyserini", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/snap-stanford/stark", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/topoteretes/cognee", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/neo4j/neo4j-graphrag-python", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/OSU-NLP-Group/HippoRAG", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/danny-avila/rag_api", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/lightonai/pylate", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/neuml/txtai", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/RUC-NLPIR/FlashRAG", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/beir-cellar/beir", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/gomate-community/TrustRAG", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/NovaSearch-Team/RAG-Retrieval", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/D-Star-AI/dsRAG", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/adithya-s-k/VARAG", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/Lightning-AI/LitServe", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/KruxAI/ragbuilder", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/AnswerDotAI/RAGatouille", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/FlagOpen/FlagEmbedding", "scope": "Infrastructure/Tool", "function": "RAG"},
        {"url": "https://github.com/superlinear-ai/raglite", "scope": "Infrastructure/Tool", "function": "RAG"},
        
        # 1A + 2D: Agentic Workflow Infrastructure (25 repos)
        {"url": "https://github.com/AgentOps-AI/agentops", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/ServiceNow/AgentLab", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/MLT-OSS/open-assistant-api", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/PrefectHQ/ControlFlow", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/microsoft/RD-Agent", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/datastax/astra-assistants-api", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/pydantic/logfire", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/crewAIInc/crewAI", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/livekit/agents", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/entropy-research/Devon", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/agent0ai/agent-zero", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/dbos-inc/dbos-transact-py", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/PrefectHQ/prefect", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/FoundationAgents/MetaGPT", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/Josh-XT/AGiXT", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/griptape-ai/griptape", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/letta-ai/letta", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/OSU-NLP-Group/TravelPlanner", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/llm-workflow-engine/llm-workflow-engine", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/chrislatimer/microagent", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/aj47/100x-orchestrator", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/langroid/langroid", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/mem0ai/mem0", "scope": "Infrastructure/Tool", "function": "Agentic"},
        {"url": "https://github.com/potpie-ai/potpie", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/camel-ai/camel", "scope": "Infrastructure/Tool", "function": "Agentic"},
        
        # 1A + 2E: Evaluation & Testing Infrastructure (22 repos - removed beir duplicate)
        {"url": "https://github.com/huggingface/optimum-benchmark", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/confident-ai/deepeval", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/stanford-crfm/helm", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/huggingface/evaluate", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/embeddings-benchmark/mteb", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/open-compass/VLMEvalKit", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/microsoft/promptbench", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/pytorch/benchmark", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/Pacific-AI-Corp/langtest", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/SWE-bench/SWE-bench", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/illuin-tech/vidore-benchmark", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/Vchitect/VBench", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/qdrant/vector-db-benchmark", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/openml/automlbenchmark", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/AmenRa/ranx", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/bigcode-project/bigcode-evaluation-harness", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/mlcommons/algorithmic-efficiency", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/LiveCodeBench/LiveCodeBench", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/Giskard-AI/giskard-oss", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/evalplus/evalplus", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        {"url": "https://github.com/IAAR-Shanghai/UHGEval", "scope": "Infrastructure/Tool", "function": "Evaluation"},
        
        # 1B + 2B: Serving & Inference Applications (12 repos)
        {"url": "https://github.com/dKosarevsky/AI-Talks", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/father-bot/chatgpt_telegram_bot", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/evilpan/gptcli", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/GaiZhenbiao/ChuanhuChatGPT", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/smaranjitghose/HiOllama", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/Vikranth3140/Mental-Health-Support-Chatbot", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/SUP3RMASS1VE/Ovis2-8B-", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/RAHB-REALTORS-Association/chat2gpt", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/devtayyabsajjad/Sehat-Connect", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/RobertoCorti/gptravel", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/LegendZDY/AI-IELTS-Speaking", "scope": "End-User Application", "function": "Serving"},
        {"url": "https://github.com/adamyodinsky/TerminalGPT", "scope": "End-User Application", "function": "Serving"},
        
        # 1B + 2C: RAG Applications (23 repos)
        {"url": "https://github.com/Quivr/quivr", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/Mintplex-Labs/anything-llm", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/imartinez/privateGPT", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/open-webui/open-webui", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/janhq/jan", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/lobehub/lobe-chat", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/qwersyk/Newelle", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/nalgeon/pokitoki", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/weaviate/Verba", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/HKUDS/Auto-Deep-Research", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/bhaskatripathi/pdfGPT", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/Cinnamon/kotaemon", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/PromtEngineer/localGPT", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/InternLM/HuixiangDou", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/h2oai/h2ogpt", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/Haste171/langchain-chatbot", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/swirlai/swirl-search", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/yvann-ba/Robby-chatbot", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/marianamartiyns/ChatBot-MunicipAI", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/alphasecio/chainlit", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/BlueBash/postgres-chatbot", "scope": "End-User Application", "function": "RAG"},
        {"url": "https://github.com/kaarthik108/snowChat", "scope": "End-User Application", "function": "RAG"},
        
        # 1B + 2D: Agentic Applications (23 repos - removed duplicates)
        {"url": "https://github.com/robusta-dev/holmesgpt", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/Armur-Ai/Auto-Pentest-GPT-AI", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/Significant-Gravitas/AutoGPT", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/notebook-intelligence/notebook-intelligence", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/existence-master/Sentient", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/langchain-ai/executive-ai-assistant", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/yossifibrahem/FIX-LLM", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/zitterbewegung/securday", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/browser-use/web-ui", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/DawoodTouseef/JARVIS", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/Josephrp/LablabAutogen", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/benjichat/langgraph-home-assistant", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/DennisDRX/Faraday-Web-Researcher-Agent", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/ryo-ma/gpt-assistants-api-ui", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/datarobot-community/predictive-content-generator", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/ricklamers/shell-ai", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/devopshobbies/devops-gpt", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/seratch/ChatGPT-in-Slack", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/kaymen99/Upwork-AI-jobs-applier", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/rusiaaman/wcgw", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/jekalmin/extended_openai_conversation", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/assafelovic/gpt-researcher", "scope": "End-User Application", "function": "Agentic"},
        {"url": "https://github.com/oobabooga/text-generation-webui", "scope": "End-User Application", "function": "Agentic"}
    ]
    
    return repositories

def save_complete_repository_input(repositories: List[Dict], filename: str = "complete_165_repositories_clean.json"):
    """Save the complete repository input to a JSON file - ARCHITECTURE REMOVED"""
    
    # Calculate distributions (removed architecture)
    scope_dist = {}
    func_dist = {}
    
    for repo in repositories:
        scope_dist[repo['scope']] = scope_dist.get(repo['scope'], 0) + 1
        func_dist[repo['function']] = func_dist.get(repo['function'], 0) + 1
    
    # Create comprehensive dataset metadata (removed architecture references)
    dataset_info = {
        "metadata": {
            "total_repositories": len(repositories),
            "creation_date": "2024-08-31",
            "classification_scheme": {
                "scope": ["Infrastructure/Tool", "End-User Application"],
                "function": ["Training", "Serving", "RAG", "Agentic", "Evaluation"]
            },
            "source": "Complete manual validation and classification - 8 categories, architecture column removed",
            "research_focus": "Technical debt in LLM deployment and orchestration systems"
        },
        "category_breakdown": {
            "1A_Infrastructure_Training": len([r for r in repositories if r["scope"] == "Infrastructure/Tool" and r["function"] == "Training"]),
            "1A_Infrastructure_Serving": len([r for r in repositories if r["scope"] == "Infrastructure/Tool" and r["function"] == "Serving"]),
            "1A_Infrastructure_RAG": len([r for r in repositories if r["scope"] == "Infrastructure/Tool" and r["function"] == "RAG"]),
            "1A_Infrastructure_Agentic": len([r for r in repositories if r["scope"] == "Infrastructure/Tool" and r["function"] == "Agentic"]),
            "1A_Infrastructure_Evaluation": len([r for r in repositories if r["scope"] == "Infrastructure/Tool" and r["function"] == "Evaluation"]),
            "1B_Application_Serving": len([r for r in repositories if r["scope"] == "End-User Application" and r["function"] == "Serving"]),
            "1B_Application_RAG": len([r for r in repositories if r["scope"] == "End-User Application" and r["function"] == "RAG"]),
            "1B_Application_Agentic": len([r for r in repositories if r["scope"] == "End-User Application" and r["function"] == "Agentic"])
        },
        "distribution_analysis": {
            "scope_distribution": scope_dist,
            "function_distribution": func_dist
        },
        "repositories": repositories
    }
    
    with open(filename, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Saved {len(repositories)} classified repositories to {filename}")
    
    # Print comprehensive summary (removed architecture distribution)
    print(f"\nComplete Dataset Summary (165 repositories) - Architecture Removed:")
    print(f"=" * 65)
    
    print(f"\nScope Distribution:")
    for scope, count in scope_dist.items():
        percentage = (count / len(repositories)) * 100
        print(f"   {scope}: {count} repositories ({percentage:.1f}%)")
    
    print(f"\nFunction Distribution:")
    for func, count in func_dist.items():
        percentage = (count / len(repositories)) * 100
        print(f"   {func}: {count} repositories ({percentage:.1f}%)")
    
    print(f"\nDetailed Category Breakdown:")
    for category, count in dataset_info['category_breakdown'].items():
        if count > 0:
            print(f"   {category.replace('_', ' ')}: {count} repositories")
    
    print(f"\nResearch Focus:")
    print(f"   • Infrastructure vs Applications: {scope_dist.get('Infrastructure/Tool', 0)} vs {scope_dist.get('End-User Application', 0)}")
    print(f"   • Largest function category: {max(func_dist, key=func_dist.get)} ({func_dist[max(func_dist, key=func_dist.get)]} repos)")
    print(f"   • Only scope + function classifications included")
    
    return dataset_info

def validate_complete_dataset(repositories: List[Dict]):
    """Comprehensive validation - ARCHITECTURE REMOVED"""
    print("Validating complete 165-repository dataset (architecture removed)...")
    
    issues = []
    
    # Check for required fields (removed architecture)
    required_fields = ['url', 'scope', 'function']
    for i, repo in enumerate(repositories):
        for field in required_fields:
            if field not in repo or not repo[field]:
                issues.append(f"Repository {i+1}: missing {field}")
    
    # Check for valid values (removed architecture validation)
    valid_scopes = ["Infrastructure/Tool", "End-User Application"]
    valid_functions = ["Training", "Serving", "RAG", "Agentic", "Evaluation"]
    
    for i, repo in enumerate(repositories):
        if repo.get('scope') not in valid_scopes:
            issues.append(f"Repository {i+1}: invalid scope '{repo.get('scope')}'")
        if repo.get('function') not in valid_functions:
            issues.append(f"Repository {i+1}: invalid function '{repo.get('function')}'")
    
    # Check for duplicate URLs
    urls = [repo['url'] for repo in repositories]
    duplicates = [url for url in set(urls) if urls.count(url) > 1]
    if duplicates:
        for dup_url in duplicates:
            indices = [i+1 for i, repo in enumerate(repositories) if repo['url'] == dup_url]
            issues.append(f"Duplicate URL found: {dup_url} (repositories {indices})")
    
    if issues:
        print("Validation errors found:")
        for error in issues[:20]:
            print(f"   {error}")
        return False
    
    print("Complete dataset validation passed!")
    print(f"   {len(repositories)} repositories validated")
    print(f"   All URLs unique and properly formatted")
    print(f"   All scope and function classifications valid")
    print(f"   Architecture column successfully removed")
    return True

# Main execution
if __name__ == "__main__":
    print("Creating complete repository input from 165-repo manual validation...")
    print("ARCHITECTURE COLUMN REMOVED - Only scope and function classifications included")
    
    # Create the complete repository list
    repositories = create_complete_repository_input()
    
    print(f"Repository count verification: {len(repositories)} repositories loaded")
    
    # Validate the complete dataset
    if validate_complete_dataset(repositories):
        # Save to file
        dataset_info = save_complete_repository_input(repositories, "complete_165_repositories_clean.json")
        
        print(f"\nComplete dataset ready for extraction!")
        print(f"   Input file: complete_165_repositories_clean.json")
        print(f"   Total repositories: {len(repositories)}")
        print(f"   Classifications: Scope + Function only")
        
        # Instructions for next step
        print(f"\nNext Steps for Full Dataset Extraction:")
        print(f"   1. Set your GitHub token in the extraction script")
        print(f"   2. Load: data = json.load(open('complete_165_repositories_clean.json'))")
        print(f"   3. Extract: repositories = data['repositories']")
        print(f"   4. Run: process_repositories_batch(github_token, repositories)")
        
    else:
        print("Dataset validation failed. Please review errors before proceeding.")