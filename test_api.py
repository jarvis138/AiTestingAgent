#!/usr/bin/env python3
"""
Simple test script for the AI Testing Agent API
"""

import requests
import json

def test_predict_endpoint():
    """Test the defect prediction endpoint"""
    url = "http://localhost:8000/predict"
    
    payload = {
        "repo": "test",
        "file": "src/app/login.py",
        "features": {
            "loc": 500,
            "complexity": 15,
            "churn": 8,
            "num_devs": 3
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing predict endpoint: {e}")
        return False

def test_explain_endpoint():
    """Test the SHAP explanation endpoint"""
    url = "http://localhost:8000/explain"
    
    payload = {
        "repo": "test",
        "file": "src/app/login.py",
        "features": {
            "loc": 500,
            "complexity": 15,
            "churn": 8,
            "num_devs": 3
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing explain endpoint: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    url = "http://localhost:8000/"
    
    try:
        response = requests.get(url)
        print(f"Health Check Status: {response.status_code}")
        print(f"Health Check Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AI Testing Agent API Endpoints...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint:")
    health_ok = test_health_endpoint()
    
    # Test predict endpoint
    print("\n2. Testing Predict Endpoint:")
    predict_ok = test_predict_endpoint()
    
    # Test explain endpoint
    print("\n3. Testing Explain Endpoint:")
    explain_ok = test_explain_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Predict Endpoint: {'✅ PASS' if predict_ok else '❌ FAIL'}")
    print(f"Explain Endpoint: {'✅ PASS' if explain_ok else '❌ FAIL'}")
    
    if all([health_ok, predict_ok, explain_ok]):
        print("\n🎉 All endpoints are working correctly!")
    else:
        print("\n⚠️  Some endpoints have issues. Check the logs above.") 