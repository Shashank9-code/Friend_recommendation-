# Project Instructions for Claude Code

## Project Overview
This repository contains Graph Neural Network (GCN, GAT) models and ablation experiments for friend recommendations.

## Git & GitHub Auto-Sync Policy
- **Automatic GitHub Sync**: After completing code changes, debugging, or creating new files requested by the user, immediately commit and push the updates to GitHub.
- **Commit Format**: Use concise, clear commit messages, e.g.:
  ```bash
  git add -A
  git commit -m "feat: <description of changes>"
  git push origin main
  ```
- **Ignore Rules**: Never stage or commit virtual environment files (`.venv/`), temporary files, or credentials.

## Common Commands
- **Run Training / Experiments**: `python friend_recommendation_gcn.py`
- **Run Ablation Study**: `python ablation_study_layers.py`
- **Run Auto Push Watcher**: `./auto_push.sh`
