import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import LogsPage from "./Logs.jsx";

// ── Global styles ─────────────────────────────────────────────────────────────
const style = document.createElement("style");
style.textContent = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --font: "Inter", system-ui, sans-serif;
    --accent: #6366f1;
    --bg: #f8f8fa;
    --surface: #ffffff;
    --bubble-bg: #f0f0f5;
    --input-bg: #ffffff;
    --border: #e4e4e8;
    --tag-bg: #eef0ff;
    --text-primary: #111118;
    --text-secondary: #6b6b80;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111118;
      --surface: #1a1a24;
      --bubble-bg: #21212e;
      --input-bg: #21212e;
      --border: #2e2e3e;
      --tag-bg: #1e1e38;
      --text-primary: #f0f0f8;
      --text-secondary: #8888a8;
    }
  }

  body { background: var(--bg); }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .logs-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .logs-row:hover { background: var(--bubble-bg); }
`;
document.head.appendChild(style);
// ─────────────────────────────────────────────────────────────────────────────

// Direct-access /logs viewer; no link from the chat app, same cookie auth.
const path = window.location.pathname.replace(/\/+$/, "");
const Root = path === "/logs" ? LogsPage : App;

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Root />
  </StrictMode>
);
