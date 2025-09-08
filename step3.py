import pandas as pd
# Quick script to auto-select top 20 per category:
df = pd.read_csv('output_step2/ranked_selection_candidates.csv')

selected = []
for category in df['category'].unique():
    category_repos = df[df['category'] == category]
    top_20 = category_repos.head(20)
    top_20['selected'] = 'Yes'
    selected.append(top_20)

final_df = pd.concat(selected)
final_df.to_csv('final_160_repositories.csv', index=False)
print(f"Selected {len(final_df)} repositories (20 per category)")