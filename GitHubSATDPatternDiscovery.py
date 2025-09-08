
"""
GitHub SATD Pattern Discovery

This script analyzes GitHub issues (500) to discover Self-Admitted Technical Debt (SATD) 
patterns using OpenAI’s GPT models. It samples issues, extracts SATD indicators, 
builds detection rules, and saves results for further analysis.
"""


import json
import random
import re
import time
from typing import List, Dict, Optional
from collections import Counter, defaultdict
import openai

class GitHubSATDPatternDiscovery:
    def __init__(self, openai_api_key: str):
        """Initialize the SATD pattern discovery system"""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.results = []
        self.discovered_patterns = {}
        self.detection_rules = {}
    
    def sample_issues_for_pattern_discovery(self, data: Dict, n_sample: int = 1000) -> List[Dict]:
        """
        Sample issues stratified by repository size and issue characteristics
        """
        print("📊 Extracting and filtering issues...")
        all_issues = []
        
        # Extract all issues with metadata
        for repo in data['repositories']:
            repo_name = repo['repo_name']
            repo_size = len(repo['raw_issues'])
            
            for issue in repo['raw_issues']:
                issue['repo_name'] = repo_name
                issue['repo_size'] = repo_size
                issue['has_body'] = bool(issue['body'] and len(issue['body']) > 50)
                issue['has_comments'] = issue['comments_count'] > 0
                all_issues.append(issue)
        
        print(f"Total issues extracted: {len(all_issues)}")
        
        # Stratified sampling - Filter for issues with substantial content
        substantial_issues = [
            issue for issue in all_issues 
            if issue['has_body'] and len(issue['title'] + (issue['body'] or '')) > 100
        ]
        
        print(f"Issues with substantial content: {len(substantial_issues)}")
        
        # Sample across different characteristics
        sample = random.sample(substantial_issues, min(n_sample, len(substantial_issues)))
        
        # Add stratification info
        for issue in sample:
            if issue['repo_size'] < 50:
                issue['size_category'] = 'small'
            elif issue['repo_size'] < 200:
                issue['size_category'] = 'medium'
            else:
                issue['size_category'] = 'large'
        
        print(f"✅ Sampled {len(sample)} issues for pattern discovery")
        return sample

    def analyze_issue_for_satd_patterns(self, issue: Dict) -> Optional[Dict]:
        """
        Use GPT-4o mini to identify SATD and extract language patterns
        """
        # Combine title and body for analysis (limit length for API efficiency)
        issue_body = issue['body'][:2000] if issue['body'] else ""
        issue_content = f"Title: {issue['title']}\n\nBody: {issue_body}"
        
        prompt = f"""Analyze this GitHub issue for Self-Admitted Technical Debt (SATD).

SATD occurs when developers/maintainers acknowledge:
- Suboptimal solutions or temporary fixes
- Known limitations or problems they plan to address
- Technical shortcuts or workarounds
- Areas needing future improvement

Issue Content:
{issue_content}

If this issue contains SATD:
1. SATD_FOUND: Yes
2. EXACT_QUOTES: List the specific phrases where SATD is expressed
3. WHO_ADMITS: Who acknowledges the debt (issue author, maintainer, contributor)?
4. DEBT_TYPE: What category (configuration, performance, implementation, design)?
5. LANGUAGE_PATTERNS: What specific language patterns indicate SATD?

If NO SATD found:
1. SATD_FOUND: No
2. REASON: Brief explanation why this isn't SATD

Format as JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            return {
                'issue_number': issue['number'],
                'repo_name': issue['repo_name'],
                'analysis': result,
                'issue_url': issue['html_url'],
                'repo_size': issue['repo_size'],
                'size_category': issue['size_category']
            }
        except Exception as e:
            print(f"❌ Error analyzing issue {issue['number']}: {e}")
            return None

    def process_sample_issues(self, sample_issues: List[Dict], max_issues: int = 100) -> List[Dict]:
        """
        Process sampled issues with GPT-4o mini analysis
        """
        print(f"\n🔍 Starting SATD pattern analysis on {min(max_issues, len(sample_issues))} issues...")
        results = []
        
        for i, issue in enumerate(sample_issues[:max_issues]):
            print(f"Processing issue {i+1}/{min(max_issues, len(sample_issues))}: {issue['repo_name']}#{issue['number']}")
            
            result = self.analyze_issue_for_satd_patterns(issue)
            if result:
                results.append(result)
            
            # Add small delay to respect API rate limits
            time.sleep(0.1)
        
        print(f"✅ Completed analysis of {len(results)} issues")
        self.results = results
        return results

    def extract_satd_patterns(self, results: List[Dict]) -> Dict:
        """
        Extract common SATD patterns from LLM analysis results
        """
        print("\n📝 Extracting SATD patterns from analysis results...")
        
        patterns = {
            'satd_phrases': [],
            'acknowledgment_patterns': [],
            'debt_types': Counter(),
            'who_admits': Counter(),
            'language_indicators': [],
            'satd_count': 0,
            'total_analyzed': len(results)
        }
        
        for result in results:
            try:
                # Clean and parse LLM response
                analysis_text = result['analysis'].strip()
                # Remove code block markers if present
                analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()
                
                analysis = json.loads(analysis_text)
                
                if analysis.get('SATD_FOUND') == 'Yes':
                    patterns['satd_count'] += 1
                    
                    # Extract quoted phrases
                    if 'EXACT_QUOTES' in analysis:
                        quotes = analysis['EXACT_QUOTES']
                        if isinstance(quotes, list):
                            patterns['satd_phrases'].extend(quotes)
                        elif isinstance(quotes, str):
                            patterns['satd_phrases'].append(quotes)
                    
                    # Count debt types - handle both strings and lists
                    debt_type = analysis.get('DEBT_TYPE', 'unknown')
                    if isinstance(debt_type, list):
                        # If it's a list, take the first item or join them
                        debt_type = debt_type[0] if debt_type else 'unknown'
                    elif not isinstance(debt_type, str):
                        debt_type = str(debt_type) if debt_type else 'unknown'
                    patterns['debt_types'][debt_type] += 1
                    
                    # Count who admits - handle both strings and lists
                    who_admits = analysis.get('WHO_ADMITS', 'unknown')
                    if isinstance(who_admits, list):
                        # If it's a list, take the first item or join them
                        who_admits = who_admits[0] if who_admits else 'unknown'
                    elif not isinstance(who_admits, str):
                        who_admits = str(who_admits) if who_admits else 'unknown'
                    patterns['who_admits'][who_admits] += 1
                    
                    # Extract language patterns
                    if 'LANGUAGE_PATTERNS' in analysis:
                        patterns['language_indicators'].append(analysis['LANGUAGE_PATTERNS'])
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Error parsing result for issue {result.get('issue_number', 'unknown')}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Unexpected error processing issue {result.get('issue_number', 'unknown')}: {e}")
                continue
        
        self.discovered_patterns = patterns
        
        # Display findings
        print("\n=== 📊 SATD PATTERN DISCOVERY RESULTS ===")
        print(f"Total issues analyzed: {patterns['total_analyzed']}")
        print(f"SATD issues found: {patterns['satd_count']} ({patterns['satd_count']/patterns['total_analyzed']*100:.1f}%)")
        print(f"Total SATD phrases extracted: {len(patterns['satd_phrases'])}")
        print(f"Most common debt types: {patterns['debt_types'].most_common(5)}")
        print(f"Who admits debt: {patterns['who_admits'].most_common(5)}")

        # Sample phrases with context
        if patterns.get('satd_examples_with_context'):
            print(f"\nSample SATD phrases with context:")
            for i, example in enumerate(patterns['satd_examples_with_context'][:5]):
                print(f"\n{i+1}. Quote: \"{example['quote']}\"")
                print(f"   Issue: {example['repo_name']}#{example['issue_number']}")
                print(f"   Title: {example['issue_title']}")
                print(f"   Debt Type: {example['debt_type']}")
                print(f"   URL: {example['issue_url']}")
        
        return patterns

    def _get_issue_title_from_result(self, result: Dict) -> str:
        """Extract issue title from analysis result"""
        # Try to extract title from the original analysis or create from available data
        analysis_text = result.get('analysis', '')
        if 'Title:' in analysis_text:
            lines = analysis_text.split('\n')
            for line in lines:
                if line.strip().startswith('Title:'):
                    return line.replace('Title:', '').strip()
        return f"Issue #{result['issue_number']}"  # Fallback
    
    def _get_context_snippet_from_result(self, result: Dict, quote: str) -> str:
        """Extract surrounding context for a quote"""
        analysis_text = result.get('analysis', '')
        if 'Body:' in analysis_text:
            try:
                body_start = analysis_text.find('Body:')
                body_text = analysis_text[body_start + 5:].strip()
                # Find the quote in the body and extract surrounding context
                quote_pos = body_text.lower().find(quote.lower()[:50])  # Match first 50 chars
                if quote_pos > -1:
                    start = max(0, quote_pos - 100)
                    end = min(len(body_text), quote_pos + len(quote) + 100)
                    return "..." + body_text[start:end] + "..."
            except:
                pass
        return f"Context from {result['repo_name']}#{result['issue_number']}"

    def build_github_satd_detection_rules(self, discovered_patterns: Dict) -> Dict:
        """
        Convert discovered patterns into detection rules
        """
        print("\n🔧 Building GitHub-specific SATD detection rules...")
        
        # Extract common words/phrases from discovered SATD quotes
        all_phrases = discovered_patterns['satd_phrases']
        
        # Common SATD indicators from pattern discovery
        github_satd_keywords = set()
        acknowledgment_phrases = set()
        
        for phrase in all_phrases:
            if not phrase or not isinstance(phrase, str):
                continue
                
            phrase_lower = phrase.lower().strip()
            words = phrase_lower.split()
            
            # Look for common SATD indicators
            satd_indicators = [
                'known issue', 'will fix', 'todo', 'temporary', 'workaround', 
                'limitation', 'hack', 'quick fix', 'needs improvement',
                'not ideal', 'suboptimal', 'technical debt'
            ]
            
            if any(indicator in phrase_lower for indicator in satd_indicators):
                github_satd_keywords.update([w for w in words if len(w) > 2])
            
            # Look for acknowledgment patterns
            acknowledgment_indicators = [
                'we know', 'acknowledged', 'aware of', 'will address',
                'known limitation', 'current approach', 'temporary solution'
            ]
            
            if any(ack in phrase_lower for ack in acknowledgment_indicators):
                acknowledgment_phrases.add(phrase_lower)
        
        detection_rules = {
            'keyword_rules': list(github_satd_keywords),
            'phrase_rules': list(acknowledgment_phrases),
            'debt_categories': dict(discovered_patterns['debt_types']),
            'admission_sources': dict(discovered_patterns['who_admits']),
            'stats': {
                'satd_prevalence': discovered_patterns['satd_count'] / discovered_patterns['total_analyzed'],
                'total_phrases': len(all_phrases),
                'unique_keywords': len(github_satd_keywords),
                'acknowledgment_patterns': len(acknowledgment_phrases)
            }
        }
        
        self.detection_rules = detection_rules
        
        print(f"✅ Built detection rules:")
        print(f"  - {len(github_satd_keywords)} unique keywords")
        print(f"  - {len(acknowledgment_phrases)} acknowledgment patterns")
        print(f"  - {len(detection_rules['debt_categories'])} debt categories")
        
        return detection_rules

    def apply_github_satd_detection(self, issue: Dict, detection_rules: Dict) -> Dict:
        """
        Apply discovered patterns to detect SATD in issues
        """
        issue_text = f"{issue['title']} {issue['body'] or ''}".lower()
        
        # Check for keyword matches
        keyword_matches = [
            keyword for keyword in detection_rules['keyword_rules'] 
            if keyword in issue_text and len(keyword) > 2
        ]
        
        # Check for phrase matches
        phrase_matches = [
            phrase for phrase in detection_rules['phrase_rules']
            if phrase in issue_text
        ]
        
        # Simple scoring system
        satd_score = len(keyword_matches) + (2 * len(phrase_matches))
        
        return {
            'issue_number': issue['number'],
            'repo_name': issue.get('repo_name'),
            'satd_score': satd_score,
            'is_potential_satd': satd_score >= 2,  # Threshold to be tuned
            'matched_keywords': keyword_matches,
            'matched_phrases': phrase_matches
        }

    def test_detection_rules(self, test_issues: List[Dict]) -> List[Dict]:
        """
        Test the detection rules on a sample of issues
        """
        print(f"\n🧪 Testing detection rules on {len(test_issues)} issues...")
        
        detection_results = []
        for issue in test_issues:
            result = self.apply_github_satd_detection(issue, self.detection_rules)
            detection_results.append(result)

        # Show preliminary results
        potential_satd = [r for r in detection_results if r['is_potential_satd']]
        print(f"✅ Found {len(potential_satd)} potential SATD issues out of {len(test_issues)} tested")
        print(f"Detection rate: {len(potential_satd)/len(test_issues)*100:.1f}%")
        
        return detection_results

    def save_results(self, filename_prefix: str = "github_satd"):
        """
        Save all results to files
        """
        print(f"\n💾 Saving results...")
        
        # Save analysis results
        with open(f'{filename_prefix}_analysis_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save discovered patterns
        with open(f'{filename_prefix}_discovered_patterns.json', 'w') as f:
            # Convert Counter objects to regular dicts for JSON serialization
            patterns_to_save = self.discovered_patterns.copy()
            patterns_to_save['debt_types'] = dict(patterns_to_save['debt_types'])
            patterns_to_save['who_admits'] = dict(patterns_to_save['who_admits'])
            json.dump(patterns_to_save, f, indent=2)
        
        # Save detection rules
        with open(f'{filename_prefix}_detection_rules.json', 'w') as f:
            json.dump(self.detection_rules, f, indent=2)
        
        print(f"✅ Results saved:")
        print(f"  - {filename_prefix}_analysis_results.json")
        print(f"  - {filename_prefix}_discovered_patterns.json") 
        print(f"  - {filename_prefix}_detection_rules.json")

def main():
    """
    Main execution function
    """
    # Configuration
    OPENAI_API_KEY = ""  # Replace with your API key
    DATA_FILE = "/home/niruthi/corrected_repo/raw_llm_dataset.json"   # Replace with your data file path
    SAMPLE_SIZE = 1000
    ANALYSIS_LIMIT = 500 # Start with 100 for testing, increase as needed

    print("🚀 Starting GitHub SATD Pattern Discovery")
    print("=" * 50)
    
    # Initialize the discovery system
    discoverer = GitHubSATDPatternDiscovery(OPENAI_API_KEY)
    
    try:
        # Step 1: Load data and sample issues
        print(f"📁 Loading data from {DATA_FILE}...")
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded data from {len(data['repositories'])} repositories")
        
        # Step 2: Sample issues for pattern discovery
        sample_issues = discoverer.sample_issues_for_pattern_discovery(data, SAMPLE_SIZE)
        
        # Step 3: Analyze issues with GPT-4o mini
        results = discoverer.process_sample_issues(sample_issues, ANALYSIS_LIMIT)
        
        # Step 4: Extract patterns from results
        patterns = discoverer.extract_satd_patterns(results)
        
        # Step 5: Build detection rules
        detection_rules = discoverer.build_github_satd_detection_rules(patterns)
        
        # Step 6: Test detection rules
        test_issues = sample_issues[:50]  # Test on different subset
        detection_results = discoverer.test_detection_rules(test_issues)
        
        # Step 7: Save all results
        discoverer.save_results()
        
        print("\n🎉 Pattern discovery completed successfully!")
        print(f"Ready to apply detection rules to full dataset of {data['metadata']['total_raw_issues_extracted']} issues")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{DATA_FILE}'")
        print("Please update the DATA_FILE variable with the correct path to your GitHub data.")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()


