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
| `final_160_repositories.csv` |Full 160 repos with Layer 1 + Layer 2 |

 

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



## Issues Metadata Structure

The LLMDebt dataset preserves hierarchical metadata for each extraction job, repository, and issue.  
Below is the schema overview:

    ```json
    {
      "root_structure": {
        "metadata": {
          "type": "Object",
          "description": "High-level statistics and metadata about the entire extraction job.",
          "fields": {
            "extraction_type": {"type": "String", "description": "Type of data extraction (e.g., 'raw_data_only')."},
            "extraction_date": {"type": "String", "description": "Timestamp of the extraction."},
            "total_repositories_attempted": {"type": "Integer", "description": "Total repositories targeted."},
            "successful_extractions": {"type": "Integer", "description": "Count of successfully extracted repositories."},
            "failed_extractions": {"type": "Integer", "description": "Count of failed extractions."},
            "success_rate_percent": {"type": "Float", "description": "Percentage of successful extractions."},
            "total_raw_issues_extracted": {"type": "Integer", "description": "Total number of individual issues extracted."}
          }
        },
        "schema_info": {
          "type": "Object",
          "description": "Information about the data schema's contents and intended use.",
          "fields": {
            "description": {"type": "String", "description": "Brief description of the raw data content."},
            "user_analysis_note": {"type": "String", "description": "Note on expected user analysis (e.g., technical debt classification)."}
          }
        },
        "repositories": {
          "type": "Array of Objects",
          "description": "A list of repositories and their extracted issues.",
          "item_structure_ref": "repository_structure"
        }
      },
      "repository_structure": {
        "type": "Object",
        "description": "Metadata and raw issues for a single GitHub repository.",
        "fields": {
          "repo_name": {"type": "String", "description": "Repository owner/name."},
          "url": {"type": "String", "description": "GitHub URL."},
          "classification_scope": {"type": "String", "description": "General domain classification (e.g., 'Infrastructure/Tool')."},
          "classification_function": {"type": "String", "description": "Project function classification (e.g., 'Training')."},
          "stars": {"type": "Integer", "description": "GitHub stars count."},
          "forks": {"type": "Integer", "description": "GitHub forks count."},
          "open_issues": {"type": "Integer", "description": "Current number of open issues."},
          "repo_age_months": {"type": "Float", "description": "Age of the repository in months."},
          "total_commits": {"type": "Integer", "description": "Total commit count."},
          "total_releases": {"type": "Integer", "description": "Total releases count."},
          "total_contributors": {"type": "Integer", "description": "Total unique contributors."},
          "active_contributors_3m": {"type": "Integer", "description": "Active contributors in the last three months."},
          "total_loc": {"type": "Integer", "description": "Total lines of code."},
          "language_breakdown": {"type": "Object", "description": "Percentage breakdown of programming languages."},
          "file_count": {"type": "Integer", "description": "Total file count."},
          "total_issues_analyzed": {"type": "Integer", "description": "Total issues collected from this repo."},
          "closed_issues_count": {"type": "Integer", "description": "Total closed issues."},
          "issue_closure_rate": {"type": "Float", "description": "Ratio of closed to total issues."},
          "commits_per_month_avg": {"type": "Float", "description": "Average commits per month."},
          "maintenance_load": {"type": "Float", "description": "Metric of maintenance effort."},
          "contributor_vitality": {"type": "Float", "description": "Metric of contributor activity."},
          "avg_loc_per_file": {"type": "Float", "description": "Average lines of code per file."},
          "raw_issues": {
            "type": "Array of Objects",
            "description": "A list of individual GitHub issues captured in raw format.",
            "item_structure_ref": "issue_structure"
          },
          "data_quality_score": {"type": "Float", "description": "Score/rating of the data quality for this repository."}
        }
      },
      "issue_structure": {
        "type": "Object",
        "description": "Raw data for a single GitHub issue.",
        "fields": {
          "number": {"type": "Integer", "description": "The issue number."},
          "title": {"type": "String", "description": "The issue title."},
          "body": {"type": "String", "description": "The full body/description of the issue."},
          "state": {"type": "String", "description": "The state of the issue ('open' or 'closed')."},
          "created_at": {"type": "String", "description": "Timestamp when the issue was created."},
          "closed_at": {"type": "String/Null", "description": "Timestamp when the issue was closed, or null if open."},
          "updated_at": {"type": "String", "description": "Timestamp of the last update to the issue."},
          "labels": {"type": "Array of Strings", "description": "List of applied labels (e.g., 'bug', 'enhancement')."},
          "assignees": {"type": "Array of Strings", "description": "List of GitHub usernames assigned to the issue."},
          "milestone": {"type": "String/Null", "description": "The milestone associated with the issue."},
          "comments_count": {"type": "Integer", "description": "Number of comments on the issue."},
          "reactions": {
            "type": "Object",
            "description": "Counts of various reactions on the issue (e.g., '+1', 'heart').",
            "fields": {
              "url": {"type": "String"},
              "total_count": {"type": "Integer"},
              "+1": {"type": "Integer"},
              "-1": {"type": "Integer"},
              "laugh": {"type": "Integer"},
              "hooray": {"type": "Integer"},
              "confused": {"type": "Integer"},
              "heart": {"type": "Integer"},
              "rocket": {"type": "Integer"},
              "eyes": {"type": "Integer"}
            }
          },
          "author": {"type": "String", "description": "The GitHub username of the issue author."},
          "html_url": {"type": "String", "description": "The direct URL to the issue."}
        }
      }
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


