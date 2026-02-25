---
title: ShinzoGPT
sdk: docker
app_port: 7860
---

# ShinzoGPT

This Space runs a Streamlit frontend (`chatbot.py`) and a FastAPI backend (`api.py`) in one Docker container.

## GitHub Actions Auto Deploy

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-to-hf-space.yml`.
On every push to `main`/`master`, it pushes the exact commit to your Hugging Face Space.

Required GitHub repository secrets:

- `HF_TOKEN`: Hugging Face User Access Token with `write` permission.
- `HF_SPACE_ID`: Full Space id in the format `username/space-name`.
