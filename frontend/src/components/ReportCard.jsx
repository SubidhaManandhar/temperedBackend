export default function ReportCard({ data }) {
  const isTampered = String(data?.stageA_label || "").toLowerCase() === "tampered";

  return (
    <section className="panel">
      <div className="sectionTop">
        <div>
          <h2>Analysis Report</h2>
          <p>Compact summary of the model decision and detected evidence.</p>
        </div>
      </div>

      <div className="reportGrid">
        <div className="reportCard highlight">
          <span className="reportLabel">Authenticity</span>
          <h3>{String(data.stageA_label).toUpperCase()}</h3>
          <p>{Number(data.stageA_confidence).toFixed(2)}%</p>
        </div>

        <div className="reportCard">
          <span className="reportLabel">Predicted Manipulation Type</span>
          <h3>{isTampered ? String(data.label).replaceAll("_", " ").toUpperCase() : "AUTHENTIC"}</h3>
          <p>{isTampered ? `${Number(data.confidence).toFixed(2)}%` : "Not applicable"}</p>
        </div>

        <div className="reportCard">
          <span className="reportLabel">Localization Layer</span>
          <h3>{data.best_layer || "N/A"}</h3>
          <p>
            {data.bbox
              ? `x=${data.bbox.x}, y=${data.bbox.y}, w=${data.bbox.w}, h=${data.bbox.h}`
              : "No suspicious region detected"}
          </p>
        </div>
      </div>
    </section>
  );
}