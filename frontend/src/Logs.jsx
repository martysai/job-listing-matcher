import { useCallback, useEffect, useState } from "react";
import { Login } from "./components/Login";
import { useAuth } from "./hooks/useAuth";

// ── Workflow presentation ─────────────────────────────────────────────────────
const WORKFLOWS = {
  chat: { label: "Chats", icon: "💬", hint: "User chat sessions" },
  scraper: { label: "Adzuna scraper", icon: "🛰", hint: "Vacancy scrape & index runs" },
  extraction: { label: "Extractions", icon: "🧪", hint: "LLM field-extraction calls" },
  system: { label: "System events", icon: "⚙", hint: "Everything else" },
};

const LEVEL_COLOR = {
  error: "#ef4444",
  warning: "#f59e0b",
  warn: "#f59e0b",
  info: "var(--text-secondary)",
  debug: "#8888a8",
};

function fmtTime(ts) {
  if (ts == null) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, {
    month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function fmtDuration(a, b) {
  if (a == null || b == null) return "";
  const s = Math.max(0, b - a);
  if (s < 1) return "<1s";
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

async function getJSON(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// Drop null / empty values so the expanded record reads cleanly.
function pruneNulls(obj) {
  if (Array.isArray(obj)) return obj.map(pruneNulls);
  if (obj && typeof obj === "object") {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      if (v === null || v === undefined) continue;
      out[k] = pruneNulls(v);
    }
    return out;
  }
  return obj;
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function LogsPage() {
  const { authed, login } = useAuth();
  if (authed === null) return null;
  if (!authed) return <Login onLogin={login} />;
  return <LogsViewer />;
}

function LogsViewer() {
  const [workflow, setWorkflow] = useState(null); // selected workflow key
  const [session, setSession] = useState(null); // selected session object
  const [workflows, setWorkflows] = useState(null);
  const [sessions, setSessions] = useState(null);
  const [events, setEvents] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0); // bumped to force a reload

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  // Top level: workflow buckets (also re-fetched on refresh).
  useEffect(() => {
    let alive = true;
    setError(""); setLoading(true);
    getJSON("/api/logs/workflows")
      .then((d) => { if (alive) setWorkflows(d); })
      .catch((e) => { if (alive) setError(String(e.message || e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [nonce]);

  // Load sessions when a workflow is opened.
  useEffect(() => {
    if (!workflow) return;
    let alive = true;
    setSessions(null); setError(""); setLoading(true);
    getJSON(`/api/logs/sessions?workflow=${encodeURIComponent(workflow)}&limit=100`)
      .then((d) => { if (alive) setSessions(d); })
      .catch((e) => { if (alive) setError(String(e.message || e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [workflow, nonce]);

  // Load events when a session is opened.
  useEffect(() => {
    if (!workflow || !session) return;
    let alive = true;
    setEvents(null); setError(""); setLoading(true);
    const p = new URLSearchParams({ workflow, limit: "500" });
    if (session.session_id != null) p.set("session_id", session.session_id);
    if (session.start_ts != null) p.set("start", String(session.start_ts));
    if (session.end_ts != null) p.set("end", String(session.end_ts));
    getJSON(`/api/logs/events?${p.toString()}`)
      .then((d) => { if (alive) setEvents(d); })
      .catch((e) => { if (alive) setError(String(e.message || e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [workflow, session, nonce]);

  return (
    <div style={S.page}>
      <Header
        crumbs={[
          { label: "Logs", onClick: () => { setWorkflow(null); setSession(null); } },
          workflow && {
            label: WORKFLOWS[workflow]?.label ?? workflow,
            onClick: () => setSession(null),
          },
          session && { label: sessionLabel(workflow, session) },
        ].filter(Boolean)}
        onRefresh={refresh}
        loading={loading}
      />

      <div style={S.body}>
        {error && <div style={S.error}>Failed to load: {error}</div>}

        {!workflow && (
          <WorkflowGrid
            workflows={workflows}
            loading={loading}
            onPick={(w) => { setWorkflow(w); setSession(null); }}
          />
        )}

        {workflow && !session && (
          <SessionList
            workflow={workflow}
            sessions={sessions}
            loading={loading}
            onPick={setSession}
          />
        )}

        {workflow && session && (
          <EventTable events={events} loading={loading} />
        )}
      </div>
    </div>
  );
}

function sessionLabel(workflow, s) {
  if (s.session_id) return s.session_id;
  return `${WORKFLOWS[workflow]?.label ?? workflow} run · ${fmtTime(s.start_ts)}`;
}

// ── Header / breadcrumbs ────────────────────────────────────────────────────────
function Header({ crumbs, onRefresh, loading }) {
  return (
    <div style={S.header}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        {crumbs.map((c, i) => (
          <span key={i} style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            {i > 0 && <span style={{ color: "var(--text-secondary)" }}>/</span>}
            <button
              onClick={c.onClick}
              disabled={!c.onClick}
              title={c.label}
              style={{
                ...S.crumb,
                cursor: c.onClick ? "pointer" : "default",
                color: i === crumbs.length - 1 ? "var(--text-primary)" : "var(--text-secondary)",
                fontWeight: i === crumbs.length - 1 ? 700 : 500,
              }}
            >
              {c.label}
            </button>
          </span>
        ))}
      </div>
      <button onClick={onRefresh} disabled={loading} title="Refresh" style={S.refreshBtn}>
        {loading ? "…" : "↻ Refresh"}
      </button>
    </div>
  );
}

// ── Level 1: workflow buckets ───────────────────────────────────────────────────
function WorkflowGrid({ workflows, loading, onPick }) {
  if (workflows == null) return <Placeholder loading={loading} empty="No logs yet." />;
  const byKey = Object.fromEntries(workflows.map((w) => [w.workflow, w]));
  const order = ["chat", "scraper", "extraction", "system"];
  const keys = [...order.filter((k) => byKey[k]), ...workflows.map((w) => w.workflow).filter((k) => !order.includes(k))];
  if (keys.length === 0) return <Placeholder empty="No logs yet." />;

  return (
    <div style={S.grid}>
      {keys.map((k) => {
        const w = byKey[k];
        const meta = WORKFLOWS[k] ?? { label: k, icon: "•", hint: "" };
        return (
          <button key={k} onClick={() => onPick(k)} style={S.card} className="logs-card">
            <div style={{ fontSize: 26, marginBottom: 8 }}>{meta.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>{meta.label}</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 2 }}>{meta.hint}</div>
            <div style={{ marginTop: 14, display: "flex", gap: 14, fontSize: 13 }}>
              <span><b>{w.count}</b> events</span>
              {w.error_count > 0 && <span style={{ color: "#ef4444" }}><b>{w.error_count}</b> errors</span>}
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
              last {fmtTime(w.last_ts)}
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ── Level 2: session list ───────────────────────────────────────────────────────
function SessionList({ workflow, sessions, loading, onPick }) {
  if (sessions == null) return <Placeholder loading={loading} empty="No sessions." />;
  if (sessions.length === 0) return <Placeholder empty="No sessions in this workflow." />;
  return (
    <div style={S.list}>
      <div style={{ ...S.sessRow, ...S.listHead }}>
        <span>Session</span>
        <span style={S.num}>Events</span>
        <span style={S.num}>Errors</span>
        <span>Started</span>
        <span style={S.num}>Duration</span>
      </div>
      {sessions.map((s, i) => (
        <button key={i} onClick={() => onPick(s)} style={S.sessRow} className="logs-row">
          <span style={S.ellip} title={sessionLabel(workflow, s)}>{sessionLabel(workflow, s)}</span>
          <span style={S.num}>{s.count}</span>
          <span style={{ ...S.num, color: s.error_count ? "#ef4444" : "inherit" }}>{s.error_count || ""}</span>
          <span style={{ color: "var(--text-secondary)" }}>{fmtTime(s.start_ts)}</span>
          <span style={{ ...S.num, color: "var(--text-secondary)" }}>{fmtDuration(s.start_ts, s.end_ts)}</span>
        </button>
      ))}
    </div>
  );
}

// ── Level 3: event table with expandable rows ───────────────────────────────────
function EventTable({ events, loading }) {
  const [open, setOpen] = useState(() => new Set());
  if (events == null) return <Placeholder loading={loading} empty="No events." />;
  if (events.length === 0) return <Placeholder empty="No events in this session." />;

  const toggle = (i) => setOpen((prev) => {
    const next = new Set(prev);
    next.has(i) ? next.delete(i) : next.add(i);
    return next;
  });

  return (
    <div style={S.eventTable}>
      <div style={{ ...S.evRow, ...S.listHead }}>
        <span />
        <span>Time</span>
        <span>Level</span>
        <span>Kind</span>
        <span>Summary</span>
      </div>
      {events.map((e, i) => {
        const isOpen = open.has(i);
        const kind = e.event || e.component || (e.logger ?? "");
        return (
          <div key={i} style={S.evGroup}>
            <button onClick={() => toggle(i)} style={S.evRow} className="logs-row">
              <span style={{ color: "var(--text-secondary)", width: 14, textAlign: "center" }}>
                {isOpen ? "▾" : "▸"}
              </span>
              <span style={S.evCell}>{fmtTime(e.ts)}</span>
              <span style={{ ...S.evCell, color: LEVEL_COLOR[e.level] ?? "inherit", fontWeight: 600 }}>
                {e.level ?? "—"}
              </span>
              <span style={S.evCellEllip} title={kind}>{kind || "—"}</span>
              <span style={S.evCellEllip} title={e.summary ?? ""}>{e.summary ?? ""}</span>
            </button>
            {isOpen && (
              <pre style={S.json}>{JSON.stringify(pruneNulls(e.raw ?? {}), null, 2)}</pre>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Placeholder({ loading, empty }) {
  return (
    <div style={{ padding: 40, textAlign: "center", color: "var(--text-secondary)", fontSize: 14 }}>
      {loading ? "Loading…" : empty}
    </div>
  );
}

// ── Styles ──────────────────────────────────────────────────────────────────────
const mono = '"SF Mono", "Cascadia Code", Consolas, "Liberation Mono", monospace';

const S = {
  page: {
    height: "100dvh", display: "flex", flexDirection: "column",
    background: "var(--bg)", color: "var(--text-primary)", fontFamily: "var(--font)",
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 22px", borderBottom: "1px solid var(--border)",
    background: "var(--surface)", gap: 12,
  },
  crumb: {
    border: "none", background: "transparent", fontFamily: "inherit",
    fontSize: 15, padding: 0, maxWidth: 420, overflow: "hidden",
    textOverflow: "ellipsis", whiteSpace: "nowrap",
  },
  refreshBtn: {
    padding: "6px 12px", borderRadius: 8, border: "1px solid var(--border)",
    background: "transparent", color: "var(--text-secondary)",
    fontSize: 13, fontFamily: "inherit", cursor: "pointer", flexShrink: 0,
  },
  body: { flex: 1, overflowY: "auto", padding: 22 },
  error: {
    padding: "10px 14px", borderRadius: 8, marginBottom: 16,
    background: "rgba(239,68,68,0.1)", color: "#ef4444", fontSize: 13,
  },
  grid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
    gap: 16,
  },
  card: {
    textAlign: "left", border: "1px solid var(--border)", borderRadius: 14,
    background: "var(--surface)", padding: "18px 20px", cursor: "pointer",
    fontFamily: "inherit", color: "inherit", transition: "border-color 0.15s, transform 0.1s",
  },
  list: {
    border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden",
    background: "var(--surface)",
  },
  sessRow: {
    display: "grid",
    gridTemplateColumns: "1fr 90px 80px 180px 90px",
    alignItems: "center", gap: 12, padding: "11px 16px", width: "100%",
    border: "none", borderBottom: "1px solid var(--border)", background: "transparent",
    color: "inherit", fontFamily: "inherit", fontSize: 14, cursor: "pointer",
    textAlign: "left",
  },
  listHead: {
    fontSize: 12, fontWeight: 700, textTransform: "uppercase",
    letterSpacing: 0.5, color: "var(--text-secondary)", cursor: "default",
    background: "var(--bg)",
  },
  num: { textAlign: "right", fontVariantNumeric: "tabular-nums" },
  ellip: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  eventTable: {
    border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden",
    background: "var(--surface)", fontFamily: mono, fontSize: 13,
  },
  evGroup: { borderBottom: "1px solid var(--border)" },
  evRow: {
    display: "grid",
    gridTemplateColumns: "20px 150px 70px 200px 1fr",
    alignItems: "center", gap: 12, padding: "8px 14px", width: "100%",
    border: "none", background: "transparent", color: "inherit",
    fontFamily: "inherit", fontSize: 13, cursor: "pointer", textAlign: "left",
  },
  evCell: { whiteSpace: "nowrap" },
  evCellEllip: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  json: {
    margin: 0, padding: "12px 16px 16px 46px", background: "var(--bg)",
    fontFamily: mono, fontSize: 12, lineHeight: 1.5, color: "var(--text-primary)",
    overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word",
  },
};
