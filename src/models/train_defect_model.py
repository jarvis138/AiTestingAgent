import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# Create synthetic dataset for demo purposes
def generate_synthetic_data(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'loc': rng.randint(10, 1000, size=n),
        'complexity': rng.randint(1, 50, size=n),
        'churn': rng.randint(0, 20, size=n),
        'num_devs': rng.randint(1, 10, size=n)
    })
    # simple function to decide defect probability
    prob = 1/(1 + np.exp(-0.005*(df['loc'] - 300) + 0.1*(df['complexity'] - 10) + 0.2*(df['churn'] - 5)))
    df['label'] = (rng.rand(n) < prob).astype(int)
    return df

def train_and_save(path='src/models/model.pkl'):
    df = generate_synthetic_data(2000)
    X = df[['loc','complexity','churn','num_devs']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to', path)

def train_kc1_model(path='src/models/defect_model.pkl'):
    """Train and save KC1-style model with feature columns metadata"""
    print("Training KC1-style defect prediction model...")
    
    # Generate larger, more realistic dataset
    df = generate_synthetic_data(5000, seed=42)
    X = df[['loc','complexity','churn','num_devs']]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, preds))
    
    # Save model with metadata
    model_data = {
        'model': model,
        'feature_columns': ['loc', 'complexity', 'churn', 'num_devs'],
        'feature_importance': dict(zip(['loc', 'complexity', 'churn', 'num_devs'], 
                                     model.feature_importances_)),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_type': 'RandomForest',
        'version': '1.0'
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'KC1 model saved to {path}')
    print(f'Feature importance: {model_data["feature_importance"]}')
    
    return model_data

if __name__ == '__main__':
    # Train both models
    print("=== Training Basic Model ===")
    train_and_save()
    
    print("\n=== Training KC1 Model ===")
    train_kc1_model()
