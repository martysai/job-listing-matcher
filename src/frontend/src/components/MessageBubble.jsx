export function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: "12px",
      }}
    >
      {!isUser && (
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: "50%",
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 15,
            flexShrink: 0,
            marginRight: 10,
            marginTop: 2,
          }}
        >
          ✦
        </div>
      )}

      <div
        style={{
          maxWidth: "72%",
          padding: "12px 16px",
          borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
          background: isUser ? "var(--accent)" : "var(--bubble-bg)",
          color: isUser ? "#fff" : "var(--text-primary)",
          fontSize: 15,
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        {message.content}
        {message.content === "" && (
          <span style={{ opacity: 0.5 }}>
            <BlinkingCursor />
          </span>
        )}
      </div>
    </div>
  );
}

function BlinkingCursor() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 2,
        height: "1em",
        background: "currentColor",
        animation: "blink 1s step-end infinite",
        verticalAlign: "text-bottom",
        marginLeft: 2,
      }}
    />
  );
}
