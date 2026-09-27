# Project Instructions for Claude Code

## 1. Context & Architecture Memory
- **Project Progress Tracker**: Consult [`PROGRESS.md`](file:///Users/shashankprabhakar/Downloads/MINOR_PROJECT%20copy/PROGRESS.md) to understand current state and roadmap before starting new tasks.
- **Update Progress**: When completing a task or roadmap step, check off the item in `PROGRESS.md`.

## 2. Token Conservation Rules (Critical)
- **Targeted File Inspection**: Only read files directly relevant to the current task. Do NOT recursively inspect the entire repository or load binary files (`.pt`, `.png`, `.jpg`, `.DS_Store`).
- **Concise Responses**: Provide clean, efficient answers. Avoid redundant code explanations.
- **Maintain Interfaces**: Respect existing modules in `src/` (`src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/utils/`).

## 3. Git & GitHub Auto-Sync Policy
- **Automatic Sync**: After completing code changes, tests, or bug fixes, immediately commit and push updates:
  ```bash
  git add -A
  git commit -m "<type>: <brief description>"
  git push origin main
  ```
- **Ignore Rules**: Never stage or commit virtual environment files (`.venv/`), cache, or credentials.

## 4. Key Project Commands
- **Run Training**: `python train.py`
- **Run Benchmark**: `python benchmark.py`
- **Run Auto Push Watcher**: `./auto_push.sh`
