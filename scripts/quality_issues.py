import json
import re
from collections import Counter

def load_github_data(file_path):
    """Load GitHub issues data"""
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_issues = []
        
        # Check if it's the expected format
        if 'repositories' in data:
            for repo in data['repositories']:
                if 'raw_issues' in repo:
                    for issue in repo['raw_issues']:
                        issue['repo_name'] = repo.get('repo_name', 'unknown')
                        all_issues.append(issue)
                else:
                    print(f"Warning: No 'raw_issues' found in repo: {repo.get('repo_name', 'unknown')}")
        else:
            print("Error: Data format not recognized. Expected 'repositories' key.")
            return []
        
        print(f"Successfully loaded {len(all_issues)} issues from {len(data['repositories'])} repositories")
        return all_issues
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def is_english(text):
    """Check if text is primarily in English"""
    if not text or len(text) < 10:
        return False
        
    # Count ASCII characters vs total
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text)
    
    # Count non-Latin characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    cyrillic_chars = len(re.findall(r'[\u0400-\u04ff]', text))
    non_latin = chinese_chars + cyrillic_chars
    
    # Count English words
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    
    return ascii_ratio > 0.7 and non_latin < 20 and english_words > 5

def has_substantial_length(issue):
    """Check if issue has substantial content"""
    title = issue.get('title', '') or ''
    body = issue.get('body', '') or ''
    total_length = len(f"{title} {body}")
    return total_length > 100

def is_empty_or_minimal(issue):
    """Check if issue is empty or has minimal content"""
    title = (issue.get('title', '') or '').strip()
    body = (issue.get('body', '') or '').strip()
    
    # Empty checks
    if not body or len(body) < 20:
        return True
        
    # Minimal content patterns
    minimal_patterns = [
        'no description provided', 'none', 'n/a', 'see title',
        'same as title', 'duplicate', '...', 'todo', 'tbd'
    ]
    
    if body.lower() in minimal_patterns:
        return True
        
    # Just error logs
    error_words = len(re.findall(r'(error|exception|traceback|failed)', body.lower()))
    if error_words > 3 and len(body.split()) < 50:
        return True
        
    # Generic titles with minimal body
    generic_titles = ['bug', 'error', 'issue', 'help', 'question', 'problem']
    if title.lower().strip() in generic_titles and len(body.split()) < 20:
        return True
        
    return False

def is_spam(issue):
    """Check if issue contains spam"""
    title = (issue.get('title', '') or '').lower()
    body = (issue.get('body', '') or '').lower()
    text = f"{title} {body}"
    
    spam_indicators = [
        'buy now', 'click here', 'free download', 'make money',
        'visit our website', 'contact us at', 'call now'
    ]
    
    spam_score = sum(1 for indicator in spam_indicators if indicator in text)
    return spam_score > 0

def count_words_in_issues(issues):
    """Count total words in issues"""
    total_words = 0
    for issue in issues:
        title = issue.get('title', '') or ''
        body = issue.get('body', '') or ''
        text = f"{title} {body}"
        # Count words (split by whitespace and filter empty strings)
        words = len([word for word in text.split() if word.strip()])
        total_words += words
    return total_words

def analyze_issue_quality(file_path):
    """Main analysis function"""
    
    print("GitHub Issues Quality Analysis")
    print("=" * 40)
    
    # Load data
    all_issues = load_github_data(file_path)
    total = len(all_issues)
    print(f"Total Issues: {total:,}")
    
    # Count words in all issues
    total_words = count_words_in_issues(all_issues)
    print(f"Total Words in All Issues: {total_words:,}")
    print(f"Average Words per Issue: {total_words/total:.1f}")
    print()
    
    # Filter 1: English Detection
    print("Filter 1: English Detection")
    english_issues = [issue for issue in all_issues if is_english(f"{issue.get('title', '')} {issue.get('body', '')}")]
    english_words = count_words_in_issues(english_issues)
    print(f"   English issues: {len(english_issues):,} ({len(english_issues)/total*100:.1f}%)")
    print(f"   English words: {english_words:,}")
    print(f"   Removed non-English: {total - len(english_issues):,}")
    print()
    
    # Filter 2: Minimum Length  
    print("Filter 2: Minimum Length (>100 chars)")
    length_filtered = [issue for issue in english_issues if has_substantial_length(issue)]
    length_words = count_words_in_issues(length_filtered)
    print(f"   Substantial length: {len(length_filtered):,} ({len(length_filtered)/total*100:.1f}%)")
    print(f"   Substantial words: {length_words:,}")
    print(f"   Removed too short: {len(english_issues) - len(length_filtered):,}")
    print()
    
    # Filter 3: Remove Empty
    print("Filter 3: Remove Empty/Minimal")
    substantial_issues = [issue for issue in length_filtered if not is_empty_or_minimal(issue)]
    substantial_words = count_words_in_issues(substantial_issues)
    print(f"   Non-empty issues: {len(substantial_issues):,} ({len(substantial_issues)/total*100:.1f}%)")
    print(f"   Non-empty words: {substantial_words:,}")
    print(f"   Removed empty/minimal: {len(length_filtered) - len(substantial_issues):,}")
    print()
    
    # Filter 4: Remove Spam
    print("Filter 4: Spam Detection")
    clean_issues = [issue for issue in substantial_issues if not is_spam(issue)]
    clean_words = count_words_in_issues(clean_issues)
    print(f"   Clean issues: {len(clean_issues):,} ({len(clean_issues)/total*100:.1f}%)")
    print(f"   Clean words: {clean_words:,}")
    print(f"   Removed spam: {len(substantial_issues) - len(clean_issues):,}")
    print()
    
    # Summary
    print("FILTERING SUMMARY")
    print("=" * 20)
    print(f"Original: {total:,} issues, {total_words:,} words")
    print(f"English: {len(english_issues):,} issues, {english_words:,} words")  
    print(f"Length: {len(length_filtered):,} issues, {length_words:,} words")
    print(f"Substantial: {len(substantial_issues):,} issues, {substantial_words:,} words")
    print(f"Final clean: {len(clean_issues):,} issues, {clean_words:,} words")
    print()
    print(f"RETENTION RATE: {len(clean_issues)/total*100:.1f}% issues, {clean_words/total_words*100:.1f}% words")
    print(f"ANALYSIS-READY: {len(clean_issues):,} issues, {clean_words:,} words")
    print(f"AVERAGE WORDS PER CLEAN ISSUE: {clean_words/len(clean_issues):.1f}")
    
    # Top repositories
    print("\nTOP 10 REPOSITORIES (by clean issues)")
    repo_counts = Counter()
    for issue in clean_issues:
        repo_counts[issue['repo_name']] += 1
    
    for repo, count in repo_counts.most_common(10):
        print(f"   {repo}: {count}")
    
    return len(clean_issues)

def main():
    """Main execution function"""
    DATA_FILE = "raw_llm_dataset.json"  # The file with actual issues data
    
    try:
        analyze_issue_quality(DATA_FILE)
        
    except FileNotFoundError:
        print(f"Error: Could not find file {DATA_FILE}. Please update DATA_FILE path.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
