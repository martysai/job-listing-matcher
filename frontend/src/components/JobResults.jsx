import { JobCard } from "./JobCard";

const sectionHeading = (color) => ({
  fontSize: 13,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color,
  marginBottom: 16,
});

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

  const newJobs = jobs.filter((j) => j.isNew);
  const oldJobs = jobs.filter((j) => !j.isNew);

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
      <div style={{ padding: "20px 20px 12px" }}>
        {newJobs.length > 0 && (
          <>
            <div style={sectionHeading("var(--accent)")}>
              {newJobs.length} matches found
            </div>
            {newJobs.map((job, i) => (
              <JobCard key={job.id} job={job} index={i} isNew />
            ))}
          </>
        )}
        {oldJobs.length > 0 && (
          <>
            <div style={{ ...sectionHeading("var(--text-secondary)"), marginTop: newJobs.length > 0 ? 24 : 0 }}>
              {oldJobs.length} older matches
            </div>
            {oldJobs.map((job, i) => (
              <JobCard key={job.id} job={job} index={i} isNew={false} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
