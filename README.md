---
title: StudyGPT
emoji: "🎓"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# StudyGPT

StudyGPT is a production-ready Next.js 15 tutoring app that combines:

- fast LLM chat with Groq by default
- OpenAI, Anthropic, and Gemini support through user-supplied API keys
- browser-native voice input/output with the Web Speech API
- a three-agent orchestration layer for text, visuals, and voice
- live interactive mind maps with React Flow
- animated charts with Recharts
- Mermaid fallback rendering
- local chat persistence with Zustand + localStorage

## Stack

- Next.js 15 App Router
- TypeScript
- Tailwind CSS
- shadcn-style UI primitives
- Zustand
- Framer Motion
- `@xyflow/react`
- Recharts
- Mermaid
- Groq SDK, OpenAI SDK, Anthropic SDK, Google GenAI SDK

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Copy the environment template and add any server-side fallback keys you want:

```bash
cp .env.example .env.local
```

3. Start the app:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000)

## Add API keys

StudyGPT supports two ways to provide keys:

1. Server environment variables

Use `.env.local`:

```env
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

2. Browser-stored user keys

- Open the `API Keys` button in the header.
- Paste one or more provider keys.
- Keys are stored in `localStorage` for the current browser only.
- If a browser key is missing, StudyGPT falls back to the server env var for that provider.

## Features

- Chat mode plus voice mode with a seamless top-bar toggle
- Continuous listening or push-to-talk voice flow
- Streamed tutoring responses
- Parallel text, visual, and voice agents for each prompt
- Interactive mind maps that animate node-by-node
- Click-to-explain node drill-down
- Editable node labels
- Recharts-based visual analytics
- Mermaid fallback diagrams
- PNG and SVG export for visuals
- Auto-saved chat history
- Suggested topics and saved topics
- Dark and light themes
- Keyboard shortcuts:
  - `Cmd/Ctrl + K` for new chat
  - `Cmd/Ctrl + Shift + M` for voice mode

## Important notes

- Voice input/output depends on browser support for `SpeechRecognition` and `SpeechSynthesis`.
- The default experience targets Chromium-based browsers for the best speech API coverage.
- The `/api/chat` route now streams typed multi-agent events over SSE.
- The explanation agent streams the main tutor answer while the visual and voice agents run in parallel.
- The exact tutoring system prompt lives in [`lib/system-prompt.ts`](./lib/system-prompt.ts).

## Deploy on Vercel

1. Push this project to GitHub.
2. Import the repository into [Vercel](https://vercel.com).
3. Add any provider keys you want in the Vercel project environment settings:
   - `GROQ_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`
4. Deploy.

No extra infrastructure is required. The app uses a standard App Router API route for streaming and keeps user chat history on the client.

## Deploy on Hugging Face Spaces

StudyGPT uses a server-side Next.js API route, so Hugging Face should be configured as a Docker Space.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose `Docker` as the SDK/template type.
3. Name the Space whatever you want to deploy to.
4. In the Space settings, add any runtime secrets you want the app to use:
   - `GROQ_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`
5. Push this repo to GitHub and enable the GitHub Actions workflow in `.github/workflows/ci-cd.yml`.

The included `Dockerfile` runs the app with Next.js standalone output, which is the right fit for Hugging Face Spaces.

## GitHub CI/CD For Hugging Face

The included GitHub Actions workflow does two things:

- runs `npm ci`, `npm run lint`, and `npm run build`
- automatically force-pushes `main` to your Hugging Face Space after validation succeeds

Configure these in your GitHub repository before expecting auto-deploys to work:

1. Repository secret: `HF_TOKEN`
   - create a Hugging Face write token with access to the target Space
2. Repository variable: `HF_SPACE_REPO` or `HF_SPACE_ID`
   - example: `shinzobolte/StudyGPT`

Once those are set, every push to `main` will redeploy the Hugging Face Space.

## Validation

The current codebase passes:

```bash
npm run lint
npm run build
```
