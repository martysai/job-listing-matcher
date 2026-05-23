# Frontend — Job Match Bot UI

React + Vite single-page app. Provides a chat interface that collects job preferences from the user and displays ranked job recommendations alongside the conversation.

## Requirements

- Node.js 18+
- The backend API running at `http://localhost:8000` (see [`backend/README.md`](../backend/README.md))

## Setup

```bash
# From frontend/
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
frontend/
├── index.html
├── vite.config.js
└── src/
    ├── main.jsx                  # Entry point, global styles, React root
    ├── App.jsx                   # Auth gate + root layout (chat panel + job results panel)
    ├── components/
    │   ├── Login.jsx             # Login screen (credentials → session cookie)
    │   ├── MessageBubble.jsx     # Single chat message (user or assistant)
    │   ├── ChatInput.jsx         # Text input + send button
    │   ├── JobCard.jsx           # Individual job listing card
    │   └── JobResults.jsx        # Job results list with loading/error states
    └── hooks/
        ├── useAuth.js            # Session check + login
        └── useChat.js            # SSE streaming, session history, search/jobs events
```

## How It Works

1. **Login** — `useAuth` checks for an existing session; if there is none, `Login` collects credentials and posts them to `/api/auth/login`. Once authenticated, the chat shell renders.
2. **Chat** — `useChat` streams responses from `POST /api/chat/stream` as SSE events. Text chunks are appended to the current assistant bubble in real time. The session id is persisted in `localStorage`, and prior history is hydrated from `/api/chat/history/{session_id}`.
3. **Search & results** — when the backend emits a `searching` event (after collecting enough info), the results panel opens with a loading state. The ranked jobs then arrive inline on the *same* stream as a `jobs` event — there is no separate recommend request.
4. **Layout** — the job results panel slides in alongside the chat once results are available; **New chat** resets the session via `/api/chat/reset/{session_id}`.
