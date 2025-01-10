<<<<<<< HEAD
# Navigate to your project directory
cd "C:\Users\Deepika\Desktop\Marketing Buddy\Application"

# Initialize git repository
git init

# Create .gitignore and add necessary files
echo ".env" > .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"

# Rename the default branch to main
git branch -M main

# Add remote repository (replace with your GitHub repository URL)
git remote add origin https://github.com/Sumit1205t/marketing-analytics-assistant.git

# Push to GitHub
git push -u origin main
=======
# marketing-analytics-assistant
Buddy to Marketing Head
>>>>>>> c0406c0e3d87a1447831536ec3857ed513a59f9c
