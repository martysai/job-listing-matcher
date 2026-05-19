export function JobCard({ job, index }) {
  const score = Math.round(job.match_score * 100);

  return (
    <a
      href={job.url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: "block",
        textDecoration: "none",
        color: "inherit",
        background: "var(--bubble-bg)",
        borderRadius: 14,
        padding: "18px 20px",
        marginBottom: 12,
        boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
        border: "1px solid var(--border)",
        transition: "transform 0.15s ease, box-shadow 0.15s ease",
        animationDelay: `${index * 60}ms`,
        animationName: "slideIn",
        animationDuration: "0.35s",
        animationFillMode: "both",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.12)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "";
        e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.07)";
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 16, color: "var(--text-primary)" }}>
            {job.title}
          </div>
          <div style={{ fontSize: 14, color: "var(--text-secondary)", marginTop: 2 }}>
            {job.company} · {job.location}
          </div>
        </div>

        {/* Match score badge */}
        <div
          style={{
            flexShrink: 0,
            marginLeft: 12,
            padding: "3px 10px",
            borderRadius: 20,
            fontSize: 12,
            fontWeight: 600,
            background: score >= 90 ? "#d1fae5" : score >= 75 ? "#fef3c7" : "#f3f4f6",
            color: score >= 90 ? "#065f46" : score >= 75 ? "#92400e" : "#374151",
          }}
        >
          {score}% match
        </div>
      </div>

      {/* Summary */}
      <p
        style={{
          margin: "10px 0 10px",
          fontSize: 14,
          color: "var(--text-secondary)",
          lineHeight: 1.55,
        }}
      >
        {job.summary}
      </p>

      {/* Footer: tags + salary */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {job.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              style={{
                padding: "2px 10px",
                borderRadius: 20,
                fontSize: 12,
                background: "var(--tag-bg)",
                color: "var(--accent)",
                fontWeight: 500,
              }}
            >
              {tag}
            </span>
          ))}
        </div>
        {job.salary && (
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
            {job.salary}
          </span>
        )}
      </div>
    </a>
  );
}
