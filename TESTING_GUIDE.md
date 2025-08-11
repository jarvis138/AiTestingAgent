# AI Testing Agent - Testing Guide

## 🚀 **Quick Start Testing**

### 1. **React Dashboard (Working Now!)**
```bash
cd dashboard
npm start
```
- **Status**: ✅ Running on http://localhost:3000
- **Test**: Open browser and verify all pages load

### 2. **Backend API (Needs Python)**
```bash
# Option A: Install Python 3.10+
# Option B: Use Docker (if Docker Desktop is running)
docker-compose up -d orchestrator

# Test endpoints:
curl http://localhost:8000/docs  # API documentation
curl http://localhost:8000/predict  # Defect prediction
```

### 3. **Model Training (Needs Python)**
```bash
python train_model.py
# This creates: src/models/defect_model.pkl
```

## 🧪 **End-to-End Testing Checklist**

### Phase 1: Frontend ✅
- [x] React dashboard loads
- [x] Navigation works
- [x] All pages render
- [x] Material-UI components display

### Phase 2: Backend (Needs Python)
- [ ] FastAPI server starts
- [ ] `/predict` endpoint works
- [ ] `/explain` endpoint works
- [ ] AI agent responds
- [ ] Sandbox runner executes tests

### Phase 3: Integration
- [ ] Dashboard connects to backend
- [ ] Real-time data display
- [ ] Test execution monitoring
- [ ] Defect prediction visualization

## 🔧 **Troubleshooting**

### Python Not Found
- Install Python 3.10+ from python.org
- Add to PATH environment variable
- Or use Docker: `docker-compose up -d`

### Dashboard Issues
- Clear browser cache
- Check console for errors (F12)
- Verify all dependencies: `npm install`

### Backend Issues
- Check if port 8000 is free
- Verify Python dependencies: `pip install -r requirements.txt`
- Check Docker Desktop is running

## 📊 **Expected Results**

### Dashboard Should Show:
- Navigation bar with 5 sections
- Dashboard with metrics and charts
- Placeholder content (to be replaced with real data)

### Backend Should Provide:
- FastAPI docs at /docs
- /predict endpoint for defect prediction
- /explain endpoint for SHAP explanations
- AI agent endpoints for test generation/execution

## 🎯 **Next Steps After Python Setup**
1. Train the defect prediction model
2. Start the FastAPI backend
3. Test API endpoints
4. Connect dashboard to backend
5. Implement real-time data display 