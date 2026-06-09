# Git Guide

> This file is a general-purpose Git reference. For project setup and the ML pipeline, see `web interface/SERVER_SETUP.md`.

## What to keep out of Git (project-specific)

The `.gitignore` already excludes these — do not force-add them:

| Path | Why |
|---|---|
| `data/` | Large `.npy` / `.pkl` dataset files |
| `models/` | Trained `.pth` weight files |
| `alphabet/` | Raw MP4 videos and extracted JPG frames |
| `__pycache__/` | Python bytecode |
| `.claude/` | Local AI assistant state |

If git is already tracking any of these, run:

```bash
git rm -r --cached data/ models/ alphabet/ __pycache__/ .claude/
git commit -m "Remove tracked files now in .gitignore"
```

---

## Initial Setup

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## Starting a Repository

```bash
# Initialize a new repo in the current folder
git init

# Stage all files
git add .

# First commit
git commit -m "Initial commit"

# Link to a remote GitHub repository
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

---

## Daily Workflow

```bash
# Check what changed
git status

# Stage all changes
git add .

# Stage a specific file
git add filename.py

# Commit staged changes
git commit -m "describe what you changed"

# Push to GitHub
git push
```

---

## Branches

```bash
# Create a new branch and switch to it
git checkout -b branch-name

# Switch between branches
git checkout main
git checkout branch-name

# Push branch to GitHub
git push -u origin branch-name

# List all branches
git branch
```

---

## Merging

```bash
# Merge a branch into main
git checkout main
git merge branch-name
git push

# Delete branch after merging
git branch -d branch-name                  # local
git push origin --delete branch-name       # remote
```

---

## Reverting Changes

```bash
# Undo uncommitted changes to a file
git checkout -- filename.py

# Undo last commit (keeps changes as uncommitted)
git reset HEAD~1

# Revert a specific commit (safe, keeps history)
git revert <commit-hash>
git push

# Revert a merge commit
git revert -m 1 <merge-commit-hash>
git push

# Hard reset to a previous commit (DANGEROUS - erases history)
git reset --hard <commit-hash>
git push --force
```

---

## Viewing History

```bash
# Show commit history
git log --oneline

# Show changes in a specific commit
git show <commit-hash>

# Show unstaged changes
git diff

# Show staged changes
git diff --staged
```

---

## .gitignore

Create a `.gitignore` file in the root of your project to exclude files from being tracked:

```
# Folders
.claude/
data/
models/
__pycache__/

# File types
*.pkl
*.npy
*.pt
*.pth
*.env
```

Apply it if git is already tracking those files:

```bash
git rm -r --cached .
git add .
git commit -m "Apply .gitignore"
```

---

## Common Errors

**Branch name has spaces:**
```bash
# Wrong
git checkout -b my new branch

# Correct
git checkout -b my-new-branch
```

**Merge conflict:**
```bash
# Git will mark conflicts in the file like this:
# <<<<<<< HEAD
# your code
# =======
# incoming code
# >>>>>>> branch-name

# 1. Open the file, manually fix the conflict
# 2. Stage and commit
git add .
git commit -m "Resolve merge conflict"
```

**Abort an in-progress merge:**
```bash
git merge --abort
```
