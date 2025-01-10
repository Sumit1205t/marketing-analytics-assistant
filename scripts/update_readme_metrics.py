import json
import os
from datetime import datetime

def update_readme_metrics():
    # Read the latest metrics
    log_dir = "logs"
    latest_log = max([f for f in os.listdir(log_dir) if f.endswith('.log')],
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
    
    with open(os.path.join(log_dir, latest_log)) as f:
        metrics_data = json.load(f)
    
    # Update README.md
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Find metrics section and update
    metrics_section = f"""
## Latest Metrics (Updated: {datetime.now().strftime('%Y-%m-%d')})

### ROUGE Scores
- ROUGE-L F1: {metrics_data['ROUGE-L']['f1']:.4f}
- ROUGE-2 F1: {metrics_data['ROUGE-2']['f1']:.4f}

### Perplexity
Score: {metrics_data['perplexity']:.2f}
"""
    
    # Replace existing metrics section or append
    if '## Latest Metrics' in readme_content:
        start = readme_content.find('## Latest Metrics')
        end = readme_content.find('##', start + 1)
        if end == -1:
            end = len(readme_content)
        readme_content = readme_content[:start] + metrics_section + readme_content[end:]
    else:
        readme_content += '\n' + metrics_section
    
    with open('README.md', 'w') as f:
        f.write(readme_content)

if __name__ == '__main__':
    update_readme_metrics() 