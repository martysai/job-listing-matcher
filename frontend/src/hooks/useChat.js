import { useState, useRef, useCallback, useEffect } from "react";

const WELCOME = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi! I'm your job search assistant. Tell me what kind of role you're looking for and I'll find the best matches for you. What type of job are you interested in?",
};

const SEARCH_TEXT = "Searching, please wait...";
const DONE_TEXT =
  "Here's what I could find. Message me again if you'd like to change your preferences and repeat the search.";

function getOrCreateSessionId() {
  let id = localStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("session_id", id);
  }
  return id;
}

export function useChat({ onSearching, onJobsReceived }) {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([WELCOME]);
  const [isStreaming, setIsStreaming] = useState(false);
  const readerRef = useRef(null);
  const searchMsgIdRef = useRef(null);

  // Hydrate from server whenever the session changes
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/chat/history/${sessionId}`)
      .then((r) => r.json())
      .then((history) => {
        if (cancelled) return;
        setMessages([WELCOME, ...history.map((m, i) => ({ ...m, id: String(i) }))]);
      })
      .catch(() => {}); // fall back to welcome message on any error
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const reset = useCallback(() => {
    fetch(`/api/chat/reset/${sessionId}`, { method: "POST" }).catch(() => {});
    const newId = crypto.randomUUID();
    localStorage.setItem("session_id", newId);
    setSessionId(newId);
    setMessages([WELCOME]);
  }, [sessionId]);

  const sendMessage = useCallback(
    async (userText) => {
      if (!userText.trim() || isStreaming) return;

      const userMsg = { id: crypto.randomUUID(), role: "user", content: userText };
      const assistantId = crypto.randomUUID();

      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      setIsStreaming(true);

      const history = [...messages, userMsg].map(({ role, content }) => ({
        role,
        content,
      }));

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: history, session_id: sessionId }),
        });

        const reader = res.body.getReader();
        readerRef.current = reader;
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop();

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const event = JSON.parse(line.slice(6));

            if (event.type === "text") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + event.content }
                    : m
                )
              );
            } else if (event.type === "searching") {
              const id = crypto.randomUUID();
              searchMsgIdRef.current = id;
              setMessages((prev) => [
                ...prev.filter((m) => m.id !== assistantId),
                { id, role: "assistant", content: SEARCH_TEXT },
              ]);
              onSearching();
            } else if (event.type === "jobs") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === searchMsgIdRef.current ? { ...m, content: DONE_TEXT } : m
                )
              );
              onJobsReceived(event.jobs);
            }
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: "Sorry, something went wrong. Please try again." }
                : m
            )
          );
        }
      } finally {
        setIsStreaming(false);
        readerRef.current = null;
      }
    },
    [messages, isStreaming, onSearching, onJobsReceived, sessionId]
  );

  const cancelStream = useCallback(() => {
    readerRef.current?.cancel();
  }, []);

  return { messages, sendMessage, isStreaming, cancelStream, reset };
}
