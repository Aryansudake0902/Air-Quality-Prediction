# Streamlit Deployment Guide

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE & Easiest)

**Best for**: Quick deployment, free hosting, automatic updates

#### Steps:

1. **Create a GitHub account** (if you don't have one)
   - Go to https://github.com
   - Sign up for free

2. **Create a new repository**
   - Click "New repository"
   - Name it (e.g., "air-quality-prediction")
   - Make it public (required for free tier)
   - Don't initialize with README

3. **Upload your files to GitHub**
   ```bash
   # Initialize git (if not already done)
   cd C:\Users\aryan\Downloads
   git init
   git add streamlit_app.py
   git add requirements.txt  # Create this if you don't have it
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

4. **Create requirements.txt** (if you don't have it)
   ```
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   joblib
   ```

5. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Select branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

6. **Your app will be live at**: `https://YOUR_APP_NAME.streamlit.app`

---

### Option 2: Heroku

**Best for**: More control, custom domains

#### Steps:

1. **Install Heroku CLI**
   - Download from: https://devcenter.heroku.com/articles/heroku-cli

2. **Create these files in your project folder:**

   **Procfile** (no extension):
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   **setup.sh**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

   **requirements.txt**:
   ```
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   joblib
   ```

3. **Deploy to Heroku:**
   ```bash
   heroku login
   heroku create your-app-name
   git init
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   heroku open
   ```

---

### Option 3: Docker + Any Cloud Provider

**Best for**: Maximum control, production environments

#### Steps:

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Create .dockerignore:**
   ```
   __pycache__
   *.pyc
   .git
   .env
   ```

3. **Build and run:**
   ```bash
   docker build -t streamlit-app .
   docker run -p 8501:8501 streamlit-app
   ```

4. **Deploy to cloud:**
   - AWS ECS/EC2
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform

---

### Option 4: Railway

**Best for**: Simple deployment, good free tier

#### Steps:

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Select your repository
6. Railway auto-detects Streamlit and deploys
7. Your app is live!

---

### Option 5: Render

**Best for**: Free tier with good performance

#### Steps:

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

---

## 📋 Pre-Deployment Checklist

- [ ] Create `requirements.txt` with all dependencies
- [ ] Test app locally: `streamlit run streamlit_app.py`
- [ ] Remove hardcoded paths (use relative paths)
- [ ] Remove local file dependencies or upload them
- [ ] Test with different data inputs
- [ ] Add error handling for missing files
- [ ] Optimize model loading (use caching)
- [ ] Add a README.md with instructions

---

## 🔧 Common Issues & Solutions

### Issue: App crashes on startup
**Solution**: Check `requirements.txt` has all dependencies

### Issue: Can't find files
**Solution**: Use relative paths, not absolute paths like `C:\Users\...`

### Issue: Model files too large
**Solution**: 
- Use Git LFS for large files
- Or host models on cloud storage (S3, etc.)
- Or train models on first run

### Issue: Memory errors
**Solution**: 
- Optimize data loading
- Use `@st.cache_data` for expensive operations
- Reduce batch sizes

---

## 💡 Best Practices

1. **Use environment variables** for API keys:
   ```python
   import os
   api_key = os.getenv('API_KEY', 'default_value')
   ```

2. **Cache expensive operations**:
   ```python
   @st.cache_data
   def load_model():
       # Expensive operation
       return model
   ```

3. **Handle errors gracefully**:
   ```python
   try:
       # Your code
   except Exception as e:
       st.error(f"Error: {e}")
   ```

4. **Optimize imports**:
   - Import heavy libraries only when needed
   - Use lazy loading for models

---

## 🎯 Recommended: Streamlit Community Cloud

**Why?**
- ✅ Completely free
- ✅ Zero configuration
- ✅ Automatic HTTPS
- ✅ Auto-deploys on git push
- ✅ Official Streamlit hosting
- ✅ Fast and reliable

**Quick Start:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy in 2 minutes!

---

## 📚 Additional Resources

- Streamlit Cloud: https://streamlit.io/cloud
- Streamlit Docs: https://docs.streamlit.io
- Deployment Guide: https://docs.streamlit.io/streamlit-community-cloud

