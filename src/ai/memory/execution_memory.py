import json, os
MEM_PATH = os.getenv('AGENT_MEMORY_PATH', 'data/memory.json')
def load():
    if not os.path.exists(MEM_PATH):
        return {}
    with open(MEM_PATH,'r') as f:
        return json.load(f)
def save(obj):
    os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)
    with open(MEM_PATH,'w') as f:
        json.dump(obj, f, indent=2)
def append_run(run):
    mem = load()
    runs = mem.get('runs',[])
    runs.append(run)
    mem['runs'] = runs
    # Track pass/fail and defect predictions for analytics
    if 'result' in run and isinstance(run['result'], dict):
        analytics = mem.get('analytics', {'pass':0,'fail':0,'defect_predictions':[]})
        if 'run_output' in run['result']:
            if 'PASS' in run['result']['run_output']:
                analytics['pass'] += 1
            if 'FAIL' in run['result']['run_output']:
                analytics['fail'] += 1
        if 'risk' in run['result']:
            analytics['defect_predictions'].append(run['result']['risk'])
        mem['analytics'] = analytics
    save(mem)
