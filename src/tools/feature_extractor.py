import os
from pydriller import RepositoryMining
import lizard
import json

def extract_basic_metrics(repo_path, since_commit=None, to_commit=None):
    """Extract basic file-level metrics for the latest commit(s).
    Returns a dict: { 'file_path': { 'loc': int, 'complexity': float, 'churn': int, 'num_devs': int } }"""
    metrics = {}
    authors_per_file = {}
    for commit in RepositoryMining(repo_path, from_commit=since_commit, to_commit=to_commit).traverse_commits():
        for mod in commit.modifications:
            fp = mod.new_path or mod.old_path
            if not fp:
                continue
            entry = metrics.setdefault(fp, {'loc':0,'complexity':0,'churn':0,'num_devs':0})
            # churn: added + deleted lines
            entry['churn'] = entry.get('churn',0) + (mod.added + mod.removed)
            authors = authors_per_file.setdefault(fp, set())
            authors.add(commit.author.email if commit.author else commit.author.name if commit.author else 'unknown')
            authors_per_file[fp] = authors
    # number of devs
    for fp, authors in authors_per_file.items():
        metrics[fp]['num_devs'] = len(authors)
    # compute LOC and complexity using lizard on workspace
    try:
        analysis = lizard.analyze_paths([repo_path])
        for f in analysis.function_list:
            rel = os.path.relpath(f.filename, repo_path)
            if rel in metrics:
                metrics[rel]['complexity'] = metrics[rel].get('complexity',0) + f.cyclomatic_complexity
                metrics[rel]['loc'] = metrics[rel].get('loc',0) + f.length
    except Exception as e:
        # fall back - leave complexity/loc as zero if lizard fails
        pass
    # finalize: ensure numeric types
    for fp, entry in metrics.items():
        entry['loc'] = int(entry.get('loc',0))
        entry['complexity'] = float(entry.get('complexity',0))
        entry['churn'] = int(entry.get('churn',0))
        entry['num_devs'] = int(entry.get('num_devs',0))
    return metrics

if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    args = p.parse_args()
    m = extract_basic_metrics(args.repo)
    print(json.dumps(m, indent=2))
