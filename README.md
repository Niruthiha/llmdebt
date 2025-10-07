# LLMDebt Dataset

> A large-scale curated dataset for studying **technical debt in Large Language Model (LLM) applications**.

---

## Overview

**LLMDebt** is a curated dataset of **18,286 GitHub issues** from **152 manually validated LLM application repositories**, designed to study **technical debt patterns** emerging in modern AI/LLM-based software systems.  
It enables a systematic investigation of novel maintenance challenges such as:

- **Prompt brittleness**
- **Orchestration complexity**
- **Model lock-in**
- **Economic and infrastructural debt**

---

## Key Features

- **152** manually validated LLM application repositories *(post–November 2022)*
- **18,286** GitHub issues with complete metadata
- **Two-layer taxonomic classification**:
  - *Layer 1*: Infrastructure/Tools vs End-User Applications  
  - *Layer 2*: Functional categories (Training, Serving, RAG, Agentic, Evaluation)
- **Rich metadata** — includes stars, forks, contributors, and temporal information
- Enables analysis of **software evolution** and **debt accumulation** in LLM ecosystems

---

## Dataset Structure

### 📁 Primary Data Files

| File | Description |
|------|--------------|
| `raw_llm_dataset.json` | Main dataset containing all GitHub issues (cleaned and processed) |
| `complete_158_repositories_clean.json` | Repository metadata for all validated repositories |
| [Repository classification with Layer 1 & Layer 2 annotations](https://docs.google.com/spreadsheets/d/1ALe6BvJUF6pj3avhlppAJFstcB_WBgvOeufE_xLzljk/edit?gid=546671505#gid=546671505)
 

---

## Repository Classification

### **Layer 1 – Target Audience**
| Category | Count | Percentage |
|-----------|--------|-------------|
| Infrastructure / Tools | 102 | 63.8% |
| End-User Applications | 58 | 36.2% |

### **Layer 2 – Functional Categories**
- Evaluation / Testing  
- Agentic  
- Training / Fine-tuning  
- Serving / Inference  
- Retrieval-Augmented Generation (RAG)  
- General Chatbot  

---

## Data Schema

### Issue Object Structure
    ```json
      {
        "issue_id": "integer",
        "repo_name": "string",
        "title": "string",
        "body": "string",
        "state": "open|closed",
        "created_at": "ISO 8601 timestamp",
        "updated_at": "ISO 8601 timestamp",
        "closed_at": "ISO 8601 timestamp or null",
        "author": "string",
        "labels": ["array of strings"],
        "comments_count": "integer",
        "reactions": "integer"
      }

### Repository Metadata Structure
    ```json
        {
          "repo_name": "string",
          "stars": "integer",
          "forks": "integer",
          "contributors": "integer",
          "created_at": "ISO 8601 timestamp",
          "last_commit": "ISO 8601 timestamp",
          "primary_language": "string",
          "topics": ["array of strings"],
          "layer1_classification": "Infrastructure|Application",
          "layer2_tags": "comma-separated string"
        }

## Dataset Statistics

| Metric | Value |
|--------|--------|
| **Total Repositories** | 152 |
| **Total Issues** | 18,286 |
| **Date Range** | November 2022 – October 2025 |
| **Mean Stars per Repository** | 8,595 |
| **Mean Issues per Repository** | 120.3 |

## Contact

For questions or collaboration inquiries, please:

- Open an issue in this repository, or  
- Contact **niruthiha.selvanayagam.1@ens.etsmtl.ca**

---

## Acknowledgments

We thank the **open-source community** and all contributors to the repositories included in this dataset.  
Their work provides invaluable insight into the evolving landscape of **LLM software development and maintenance**.


