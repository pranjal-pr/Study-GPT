# Security Notes

## Key handling
- Never commit real API keys.
- Use `.env` locally (already gitignored) and Hugging Face Space `Secrets` in production.
- Treat any key pasted in chat/issues/screenshots as compromised and rotate it immediately.

## Automated checks
- `Secret Scan` workflow runs on each push/PR using Gitleaks.
- CI runs lint/type/test checks before deploy workflow proceeds.
- Git history was scanned for common key patterns before `v1.0.0` release and no tracked leaks were found.

## Incident response checklist
1. Rotate exposed keys at provider dashboards (Groq/Moonshot/Hugging Face).
2. Replace secrets in local `.env`, GitHub Secrets, and HF Space Secrets.
3. Verify no secret remains in tracked files:
   - `rg -n "gsk_|nvapi-|AIza|hf_" --hidden --glob '!.git/*'`
4. If a key was committed, rewrite git history and force push.
