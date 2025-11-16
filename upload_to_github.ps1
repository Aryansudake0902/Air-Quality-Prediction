# PowerShell script to upload project to GitHub
# Make sure Git is installed first!

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Uploading to GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or use GitHub Desktop: https://desktop.github.com/" -ForegroundColor Yellow
    exit 1
}

# Navigate to project directory
$projectDir = "C:\Users\aryan\Downloads"
Set-Location $projectDir
Write-Host "Working directory: $projectDir" -ForegroundColor Green
Write-Host ""

# Initialize git if not already done
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
}

# Add files
Write-Host "Adding files..." -ForegroundColor Yellow
git add streamlit_app.py
git add requirements.txt
git add DEPLOYMENT_GUIDE.md
git add README.md
git add Procfile
git add Dockerfile
git add setup.sh
if (Test-Path ".streamlit\config.toml") {
    git add .streamlit\config.toml
}

# Check if there are changes to commit
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No changes to commit. Files may already be committed." -ForegroundColor Yellow
} else {
    Write-Host "Committing files..." -ForegroundColor Yellow
    git commit -m "Initial commit: Air Quality Prediction System"
}

# Set branch to main
Write-Host "Setting branch to main..." -ForegroundColor Yellow
git branch -M main

# Add remote (remove if exists first)
$remoteExists = git remote | Select-String -Pattern "origin"
if ($remoteExists) {
    Write-Host "Removing existing remote..." -ForegroundColor Yellow
    git remote remove origin
}

Write-Host "Adding remote repository..." -ForegroundColor Yellow
git remote add origin https://github.com/Aryansudake0902/Air-Quality-Prediction.git

# Verify remote
Write-Host ""
Write-Host "Remote repositories:" -ForegroundColor Cyan
git remote -v

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to push!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Make sure you're logged into GitHub" -ForegroundColor White
Write-Host "2. You may need a Personal Access Token (not password)" -ForegroundColor White
Write-Host "   Get one at: https://github.com/settings/tokens" -ForegroundColor White
Write-Host "3. Run: git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Or use GitHub Desktop for easier upload!" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to push now
$push = Read-Host "Do you want to push now? (y/n)"
if ($push -eq "y" -or $push -eq "Y") {
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    Write-Host ""
    Write-Host "✓ Done! Check your repository:" -ForegroundColor Green
    Write-Host "https://github.com/Aryansudake0902/Air-Quality-Prediction" -ForegroundColor Cyan
} else {
    Write-Host "Run 'git push -u origin main' when ready!" -ForegroundColor Yellow
}

