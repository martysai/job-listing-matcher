# Frontend — Job Match Bot UI

React + Vite single-page app. Provides a chat interface that collects job preferences from the user and displays ranked job recommendations alongside the conversation.

## Requirements

- Node.js 18+
- The backend API running at `http://localhost:8000` (see [`src/backend/README.md`](../backend/README.md))

## Setup

```bash
# From src/frontend/
npm install
```

## Running

```bash
# Development server with hot reload (http://localhost:5173)
npm run dev

# Production build (output in dist/)
npm run build

# Preview the production build locally
npm run preview
```

All `/api/*` requests are proxied to `http://localhost:8000` during development, so no CORS configuration is needed.

## Project Structure

```
src/frontend/
├── index.html
├── vite.config.js
└── src/
    ├── main.jsx                  # Entry point, global styles, React root
    ├── App.jsx                   # Root layout: chat panel + job results panel
    ├── components/
    │   ├── MessageBubble.jsx     # Single chat message (user or assistant)
    │   ├── ChatInput.jsx         # Text input + send button
    │   ├── JobCard.jsx           # Individual job listing card
    │   └── JobResults.jsx        # Job results list with loading/error states
    └── hooks/
        ├── useChat.js            # SSE streaming, message history management
        └── useJobs.js            # Job recommendation fetch, loading state
```

## How It Works

1. **Chat** — `useChat` streams responses from `POST /api/chat/stream` as SSE events. Text tokens are appended to the current assistant bubble in real time.
2. **Search trigger** — when the backend emits a `ready_to_search` event (after collecting enough info), `useJobs` fires `POST /api/jobs/recommend` with the extracted user profile.
3. **Layout** — the job results panel slides in alongside the chat once results are available.
