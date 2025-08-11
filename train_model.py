#!/usr/bin/env python3
"""
Standalone script to train the KC1 defect prediction model.
Run this before starting the services to ensure /predict and /explain work.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.train_defect_model import train_kc1_model

if __name__ == '__main__':
    print("Training KC1 defect prediction model...")
    try:
        model_data = train_kc1_model()
        print("\n✅ Model training completed successfully!")
        print(f"Model saved to: src/models/defect_model.pkl")
        print(f"Feature columns: {model_data['feature_columns']}")
        print(f"Model type: {model_data['model_type']}")
        print(f"Training samples: {model_data['training_samples']}")
        print("\nYou can now start the services and use /predict and /explain endpoints.")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        sys.exit(1) 