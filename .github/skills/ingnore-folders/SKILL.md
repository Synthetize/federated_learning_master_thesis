---
name: ignore-folders
description: Avoid reading virtual environment folders. Use when exploring files or searching code in this repository.
---

When inspecting the repository, never load `.venv/` (or files inside it) into context.

Rules:
- Exclude `.venv/**` from file discovery and content searches.
- Prefer reading source files under project folders, not dependencies or generated environments.
