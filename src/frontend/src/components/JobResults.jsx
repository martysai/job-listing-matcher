import { JobCard } from "./JobCard";

export function JobResults({ jobs, isLoading, error }) {
  if (isLoading) {
    return (
      <div style={{ padding: 24, textAlign: "center", color: "var(--text-secondary)" }}>
        <div style={{ fontSize: 28, marginBottom: 12, animation: "spin 1.5s linear infinite", display: "inline-block" }}>⟳</div>
        <div>Searching for your best matches…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 24, color: "#dc2626", fontSize: 14 }}>
        Error fetching jobs: {error}
      </div>
    );
  }

  if (jobs.length === 0) return null;

  return (
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
        <JobCard key={job.id} job={job} index={i} />
      ))}
    </div>
  );
}
