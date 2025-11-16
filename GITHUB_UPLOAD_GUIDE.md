# Upload Project to GitHub - Step by Step Guide

## Option 1: Install Git and Use Command Line (Recommended)

### Step 1: Install Git

1. **Download Git for Windows:**
   - Go to: https://git-scm.com/download/win
   - Download the installer
   - Run the installer (use default settings)

2. **Verify Installation:**
   - Open PowerShell or Command Prompt
   - Run: `git --version`
   - You should see the version number

### Step 2: Upload Your Project

Open PowerShell in the Downloads folder and run these commands:

```powershell
# Navigate to your project folder
cd C:\Users\aryan\Downloads

# Initialize git repository
git init

# Add all files
git add streamlit_app.py requirements.txt DEPLOYMENT_GUIDE.md README.md Procfile Dockerfile setup.sh .streamlit\config.toml

# Configure git (replace with your email)
git config user.name "Aryansudake0902"
git config user.email "your-email@example.com"

# Commit files
git commit -m "Initial commit: Air Quality Prediction System"

# Rename branch to main
git branch -M main

# Add remote repository
git remote add origin https://github.com/Aryansudake0902/Air-Quality-Prediction.git

# Push to GitHub
git push -u origin main
```

**Note:** When you run `git push`, you'll be prompted for your GitHub username and password. 
- Username: `Aryansudake0902`
- Password: Use a **Personal Access Token** (not your GitHub password)

### Step 3: Create Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Streamlit App Upload"
4. Select scopes: Check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

---

## Option 2: Use GitHub Desktop (Easier - No Command Line)

### Step 1: Install GitHub Desktop

1. Download from: https://desktop.github.com/
2. Install and sign in with your GitHub account

### Step 2: Upload Project

1. Open GitHub Desktop
2. Click "File" → "Add Local Repository"
3. Click "Choose..." and select: `C:\Users\aryan\Downloads`
4. If it says "This directory does not appear to be a Git repository":
   - Click "create a repository"
   - Name: `Air-Quality-Prediction`
   - Click "Create Repository"
5. You'll see all your files listed
6. At the bottom, type commit message: "Initial commit: Air Quality Prediction System"
7. Click "Commit to main"
8. Click "Publish repository" (top right)
9. Uncheck "Keep this code private" (if you want it public)
10. Click "Publish repository"

Done! Your code is now on GitHub.

---

## Option 3: Use GitHub Web Interface (No Installation Needed)

### Step 1: Prepare Files

Make sure all these files are in `C:\Users\aryan\Downloads`:
- streamlit_app.py
- requirements.txt
- DEPLOYMENT_GUIDE.md
- README.md
- Procfile
- Dockerfile
- setup.sh
- .streamlit/config.toml

### Step 2: Upload via Web

1. Go to: https://github.com/Aryansudake0902/Air-Quality-Prediction
2. Click "uploading an existing file" (or "Add file" → "Upload files")
3. Drag and drop all your files OR click "choose your files"
4. Scroll down, type commit message: "Initial commit: Air Quality Prediction System"
5. Click "Commit changes"

Done! Your files are uploaded.

---

## Option 4: Quick Script (After Installing Git)

I've created a script that does everything automatically. After installing Git, run:

```powershell
cd C:\Users\aryan\Downloads
.\upload_to_github.ps1
```

---

## Files to Upload

Make sure these files are included:
- ✅ streamlit_app.py (main application)
- ✅ requirements.txt (dependencies)
- ✅ README.md (documentation)
- ✅ DEPLOYMENT_GUIDE.md (deployment instructions)
- ✅ Procfile (for Heroku)
- ✅ Dockerfile (for Docker)
- ✅ setup.sh (setup script)
- ✅ .streamlit/config.toml (Streamlit config)

---

## Troubleshooting

### Issue: "git is not recognized"
**Solution:** Install Git from https://git-scm.com/download/win

### Issue: "Authentication failed"
**Solution:** Use Personal Access Token instead of password

### Issue: "Repository not found"
**Solution:** Make sure the repository exists at: https://github.com/Aryansudake0902/Air-Quality-Prediction

### Issue: "Permission denied"
**Solution:** Check that you're logged into the correct GitHub account

---

## After Uploading

Once your code is on GitHub, you can:

1. **Deploy to Streamlit Cloud:**
   - Go to: https://share.streamlit.io
   - Connect your repository
   - Deploy!

2. **Share your repository:**
   - Your repo will be at: https://github.com/Aryansudake0902/Air-Quality-Prediction
   - Share this link with others

3. **Make changes:**
   - Edit files locally
   - Commit and push changes
   - They'll appear on GitHub automatically

---

## Quick Reference

**Repository URL:** https://github.com/Aryansudake0902/Air-Quality-Prediction

**Recommended Method:** GitHub Desktop (easiest) or Git command line (most flexible)

