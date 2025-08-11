import os, pickle, numpy as np
import shap
import json

MODEL_PATH = os.getenv('MODEL_PATH', 'src/models/defect_model.pkl')

def explain_instance(features: dict):
    # Load model and feature columns
    with open(MODEL_PATH, 'rb') as f:
        loaded = pickle.load(f)
    if isinstance(loaded, dict) and 'model' in loaded and 'feature_columns' in loaded:
        model = loaded['model']
        cols = loaded['feature_columns']
    else:
        raise ValueError('Model is not in expected dict format with feature_columns')
    
    # prepare input - ensure it's a 2D array
    X = np.array([[features.get(c, 0) for c in cols]])
    
    # shap explain requires a background dataset - we use zeros or small sample; for production, supply real background
    background = np.zeros((1, len(cols)))
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(X)
    
    # format output
    out = []
    for name, val, sv in zip(cols, X[0], shap_values.values[0]):
        out.append({'feature': name, 'value': float(val), 'shap': float(sv)})
    
    return {'explanation': out, 'expected_value': float(shap_values.base_values[0])}

if __name__ == '__main__':
    # demo run
    demo = { 'loc': 400, 'complexity': 12, 'churn': 5, 'num_devs': 2 }
    print(json.dumps(explain_instance(demo), indent=2))
