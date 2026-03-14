export default function EvidenceGrid({ data }) {
  const isTampered = String(data?.stageA_label || "").toLowerCase() === "tampered";

  return (
    <section className="panel">
      <div className="sectionTop">
        <div>
          <h2>Forensic Evidence</h2>
          <p>Visual outputs used to support and explain the model decision.</p>
        </div>
      </div>

      <div className="evidenceGrid">
        <figure className="evidenceCard">
          <figcaption>
            <strong>Original Image</strong>
            <span>Uploaded evidence image</span>
          </figcaption>
          {data.original_url ? (
            <img src={data.original_url} alt="Original" />
          ) : (
            <div className="imgFallback">Not available</div>
          )}
        </figure>

        <figure className="evidenceCard">
          <figcaption>
            <strong>ELA Visualization</strong>
            <span>Highlights compression inconsistencies</span>
          </figcaption>
          {data.ela_url ? (
            <img src={data.ela_url} alt="ELA map" />
          ) : (
            <div className="imgFallback">Not available</div>
          )}
        </figure>

        <figure className="evidenceCard">
          <figcaption>
            <strong>Noise Residual Map</strong>
            <span>Shows abnormal residual patterns</span>
          </figcaption>
          {data.noise_url ? (
            <img src={data.noise_url} alt="Noise map" />
          ) : (
            <div className="imgFallback">Not available</div>
          )}
        </figure>

        <figure className="evidenceCard">
          <figcaption>
            <strong>GradCAM Heatmap</strong>
            <span>Model attention over suspicious area</span>
          </figcaption>
          {isTampered && data.heatmap_url ? (
            <img src={data.heatmap_url} alt="GradCAM heatmap" />
          ) : (
            <div className="imgFallback">Not generated for authentic images</div>
          )}
        </figure>

        <figure className="evidenceCard">
          <figcaption>
            <strong>Binary Suspicion Mask</strong>
            <span>Extracted suspicious manipulation region</span>
          </figcaption>
          {isTampered && data.mask_url ? (
            <img src={data.mask_url} alt="Binary suspicion mask" />
          ) : (
            <div className="imgFallback">Not generated for authentic images</div>
          )}
        </figure>

        <figure className="evidenceCard">
          <figcaption>
            <strong>Annotated Overlay</strong>
            <span>Localized region over the original image</span>
          </figcaption>
          {isTampered && data.heatmap_url ? (
            <img src={data.heatmap_url} alt="Annotated overlay" />
          ) : (
            <div className="imgFallback">Not generated for authentic images</div>
          )}
        </figure>
      </div>
    </section>
  );
}