import { JobCard } from "./JobCard";

function StatusLine({ jobs }) {
  const newCount = jobs.filter((j) => j.isNew).length;
  const total = jobs.length;
  if (newCount === 0) return null;
  const text =
    total > newCount
      ? `${newCount} new vacancies found, ${total} total.`
      : `${newCount} new vacancies found.`;
  return (
    <div
      style={{
        marginTop: 8,
        paddingBottom: 12,
        fontSize: 12,
        color: "var(--text-secondary)",
        opacity: 0.7,
      }}
    >
      {text}
    </div>
  );
}

export function JobResults({ jobs, isLoading }) {
  if (isLoading && jobs.length === 0) {
    return (
      <div style={{ padding: 24, textAlign: "center", color: "var(--text-secondary)" }}>
        <div style={{ fontSize: 28, marginBottom: 12, animation: "spin 1.5s linear infinite", display: "inline-block" }}>⟳</div>
        <div>Searching for your best matches…</div>
      </div>
    );
  }

  if (jobs.length === 0) return null;

  return (
    <div style={{ position: "relative" }}>
      {isLoading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ position: "absolute", inset: 0, background: "var(--surface)", opacity: 0.85 }} />
          <div style={{ position: "relative", textAlign: "center", color: "var(--text-secondary)" }}>
            <div style={{ fontSize: 28, marginBottom: 8, animation: "spin 1.5s linear infinite", display: "inline-block" }}>⟳</div>
            <div>Updating results…</div>
          </div>
        </div>
      )}
      <div style={{ padding: "20px 20px 8px" }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--accent)",
            marginBottom: 16,
          }}
        >
          {jobs.length} matches found
        </div>
        {jobs.map((job, i) => (
          <JobCard key={job.id} job={job} index={i} isNew={job.isNew} />
        ))}
        <StatusLine jobs={jobs} />
      </div>
    </div>
  );
}
