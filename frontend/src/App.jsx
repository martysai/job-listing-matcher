import { useCallback, useEffect, useRef, useState } from "react";
import { MessageBubble } from "./components/MessageBubble";
import { ChatInput } from "./components/ChatInput";
import { JobResults } from "./components/JobResults";
import { useChat } from "./hooks/useChat";

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [showJobs, setShowJobs] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  const handleSearching = useCallback(() => {
    setShowJobs(true);
    setJobsLoading(true);
  }, []);

  const handleJobsReceived = useCallback((newJobs) => {
    setJobsLoading(false);
    setJobs((prev) => [
      ...newJobs.map((j) => ({ ...j, isNew: true })),
      ...prev.map((j) => ({ ...j, isNew: false })),
    ]);
  }, []);

  const { messages, sendMessage, isStreaming, reset } = useChat({
    onSearching: handleSearching,
    onJobsReceived: handleJobsReceived,
  });

  const handleReset = useCallback(() => {
    reset();
    setShowJobs(false);
    setJobs([]);
    setJobsLoading(false);
  }, [reset]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keep scroll pinned to bottom during the panel-open layout transition
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (gap < 150) el.scrollTop = el.scrollHeight;
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      style={{
        display: "flex",
        height: "100dvh",
        background: "var(--bg)",
        fontFamily: "var(--font)",
        color: "var(--text-primary)",
      }}
    >
      {/* ── Left panel: Chat ── */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          width: showJobs ? "42%" : "100%",
          maxWidth: showJobs ? "none" : 720,
          margin: showJobs ? 0 : "0 auto",
          borderRight: showJobs ? "1px solid var(--border)" : "none",
          transition: "width 0.4s ease",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "18px 24px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            gap: 12,
            background: "var(--surface)",
          }}
        >
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: "50%",
              background: "var(--accent)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              color: "#fff",
            }}
          >
            ✦
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: 16 }}>Job Match</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              {isStreaming ? "Typing…" : "Online"}
            </div>
          </div>

          {/* Reset button */}
          <button
            onClick={handleReset}
            disabled={isStreaming}
            title="Start a new chat"
            style={{
              padding: "6px 14px",
              borderRadius: 8,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-secondary)",
              fontSize: 13,
              fontFamily: "inherit",
              cursor: isStreaming ? "not-allowed" : "pointer",
              transition: "color 0.15s, border-color 0.15s",
            }}
            onMouseEnter={(e) => {
              if (!isStreaming) {
                e.currentTarget.style.color = "var(--text-primary)";
                e.currentTarget.style.borderColor = "var(--text-secondary)";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--text-secondary)";
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          >
            New chat
          </button>
        </div>

        {/* Messages */}
        <div ref={messagesContainerRef} style={{ flex: 1, overflowY: "auto", padding: "20px 20px 8px" }}>
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSend={sendMessage} disabled={isStreaming} />
      </div>

      {/* ── Right panel: Job results ── */}
      {showJobs && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            background: "var(--surface)",
          }}
        >
          {/* Results header */}
          <div
            style={{
              padding: "18px 24px",
              borderBottom: "1px solid var(--border)",
              fontWeight: 700,
              fontSize: 16,
              background: "var(--surface)",
              position: "sticky",
              top: 0,
              zIndex: 1,
            }}
          >
            Recommended Jobs
          </div>
          <JobResults jobs={jobs} isLoading={jobsLoading} />
        </div>
      )}
    </div>
  );
}
