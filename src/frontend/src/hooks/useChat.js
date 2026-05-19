import { useState, useRef, useCallback } from "react";

/**
 * useChat — manages message history and streams responses from the backend.
 *
 * Events from the backend:
 *   { type: "text",           content: "..." }   → append to current assistant bubble
 *   { type: "ready_to_search", profile: {...} }  → trigger job search
 *   { type: "done" }                             → stream finished
 */
export function useChat({ onReadyToSearch }) {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm your job search assistant. Tell me what kind of role you're looking for and I'll find the best matches for you. What type of job are you interested in?",
    },
  ]);
  const [isStreaming, setIsStreaming] = useState(false);
  const readerRef = useRef(null);

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

      // Build history for the API (exclude the empty assistant placeholder)
      const history = [...messages, userMsg].map(({ role, content }) => ({
        role,
        content,
      }));

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history,
            session_id: "demo-session",
          }),
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
          buf = lines.pop(); // keep incomplete last line

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
            } else if (event.type === "ready_to_search") {
              onReadyToSearch(event.profile);
            }
            // "done" → loop will break via reader.read() returning done=true
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
    [messages, isStreaming, onReadyToSearch]
  );

  const cancelStream = useCallback(() => {
    readerRef.current?.cancel();
  }, []);

  return { messages, sendMessage, isStreaming, cancelStream };
}
