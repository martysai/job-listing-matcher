import { useState } from "react";

export function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        padding: "16px 20px",
        borderTop: "1px solid var(--border)",
        background: "var(--surface)",
      }}
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Tell me what you're looking for…"
        disabled={disabled}
        rows={1}
        style={{
          flex: 1,
          resize: "none",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: "10px 14px",
          fontSize: 15,
          fontFamily: "inherit",
          background: "var(--input-bg)",
          color: "var(--text-primary)",
          outline: "none",
          lineHeight: 1.5,
          transition: "border-color 0.15s",
        }}
        onFocus={(e) => (e.target.style.borderColor = "var(--accent)")}
        onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        style={{
          padding: "10px 20px",
          borderRadius: 12,
          border: "none",
          background: disabled || !value.trim() ? "var(--border)" : "var(--accent)",
          color: "#fff",
          fontSize: 15,
          fontWeight: 600,
          cursor: disabled || !value.trim() ? "not-allowed" : "pointer",
          transition: "background 0.15s",
          flexShrink: 0,
        }}
      >
        Send
      </button>
    </div>
  );
}
