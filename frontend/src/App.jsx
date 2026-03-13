import { useEffect, useRef, useState } from "react";
import "./styles.css";

const API_BASE = "";

function isAllowedImage(file) {
  if (!file) return false;

  const allowedMime = ["image/jpeg", "image/png"];
  const name = file.name.toLowerCase();

  return (
    allowedMime.includes(file.type) ||
    name.endsWith(".jpg") ||
    name.endsWith(".jpeg") ||
    name.endsWith(".png")
  );
}

export default function App() {
  const inputRef = useRef(null);

  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const isTampered = String(data?.stageA_label || "").toLowerCase() === "tampered";

  async function analyze() {
    if (!file) return;

    setBusy(true);
    setToast("Running forensic analysis...");
    setData(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: form,
      });

      const json = await res.json();

      if (!res.ok) {
        throw new Error(json.error || "Prediction failed");
      }

      setData(json);
      setToast("Analysis completed successfully.");
    } catch (e) {
      setToast(`Failed: ${String(e.message).slice(0, 180)}`);
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setFile(null);
    setPreviewUrl("");
    setData(null);
    setToast("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="siteShell">
      <section className="hero">
        <div className="heroInner">
          <p className="eyebrow">Digital Image Forensics</p>
          <h1>ForenSight</h1>
          <p className="heroText">
            A forensic analysis platform for image tampering detection using
            error level analysis, noise residual mapping, deep classification,
            and Grad-CAM localization.
          </p>
          <div className="heroActions">
            <a href="#analyze" className="primaryLinkBtn">Analyze Image</a>
            <a href="#workflow" className="secondaryLinkBtn">How It Works</a>
          </div>
        </div>
      </section>

      <main className="pageWrap">
        <section className="panel" id="analyze">
          <div className="sectionTop">
            <div>
              <h2>Submit Image for Forensic Analysis</h2>
              <p>
                Upload a JPG, JPEG, or PNG image to inspect authenticity,
                predicted manipulation type, and supporting forensic evidence.
              </p>
            </div>
          </div>

          <div
            className={`dropzone ${drag ? "dragging" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              const f = e.dataTransfer.files?.[0];

              if (f && isAllowedImage(f)) {
                setFile(f);
                setToast("");
              } else {
                setToast("Please upload a valid JPG, JPEG, or PNG image.");
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f && isAllowedImage(f)) {
                  setFile(f);
                  setToast("");
                } else if (f) {
                  setToast("Please upload a valid JPG, JPEG, or PNG image.");
                }
              }}
            />

            <div className="dropzoneText">
              <div className="dropIcon">⬆</div>
              <div>
                <h3>Drag & drop your image here</h3>
                <p>or click to browse files</p>
              </div>
            </div>

            {file && (
              <div className="uploadPreview">
                <img src={previewUrl} alt="Uploaded preview" className="uploadPreviewImage" />
                <div className="uploadMeta">
                  <strong>{file.name}</strong>
                  <span>{Math.round(file.size / 1024)} KB</span>
                  <small>Selected evidence image</small>
                </div>
              </div>
            )}
          </div>

          <div className="buttonRow">
            <button className="primaryBtn" onClick={analyze} disabled={!file || busy}>
              {busy ? "Processing..." : "Analyze"}
            </button>
            <button className="secondaryBtn" onClick={reset} disabled={!file || busy}>
              Reset
            </button>
          </div>

          {toast && <div className="statusBox">{toast}</div>}
        </section>

        <section className="panel">
          <div className="sectionTop">
            <div>
              <h2>Analysis Report</h2>
              <p>
                A compact summary of the model decision and localized evidence.
              </p>
            </div>
          </div>

          {!data ? (
            <div className="placeholderBox">
              Run an analysis to generate the report.
            </div>
          ) : (
            <>
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
                  <span className="reportLabel">Localization</span>
                  <h3>{data.best_layer || "N/A"}</h3>
                  <p>
                    {data.bbox
                      ? `x=${data.bbox.x}, y=${data.bbox.y}, w=${data.bbox.w}, h=${data.bbox.h}`
                      : "No suspicious region detected"}
                  </p>
                </div>
              </div>
            </>
          )}
        </section>

        <section className="panel">
          <div className="sectionTop">
            <div>
              <h2>Forensic Evidence</h2>
              <p>
                Visual maps used to support the model’s decision and improve interpretability.
              </p>
            </div>
          </div>

          {!data ? (
            <div className="placeholderBox">
              Generated forensic outputs will appear here after analysis.
            </div>
          ) : (
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
                  <strong>ELA Map</strong>
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
                  <strong>Noise Map</strong>
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
                  <strong>Heatmap Overlay</strong>
                  <span>Model attention over suspicious area</span>
                </figcaption>
                {isTampered && data.heatmap_url ? (
                  <img src={data.heatmap_url} alt="Heatmap overlay" />
                ) : (
                  <div className="imgFallback">Not generated for authentic images</div>
                )}
              </figure>

              <figure className="evidenceCard">
                <figcaption>
                  <strong>Binary Mask</strong>
                  <span>Extracted manipulation region</span>
                </figcaption>
                {isTampered && data.mask_url ? (
                  <img src={data.mask_url} alt="Binary mask" />
                ) : (
                  <div className="imgFallback">Not generated for authentic images</div>
                )}
              </figure>
            </div>
          )}
        </section>

        <section className="panel" id="workflow">
          <div className="sectionTop">
            <div>
              <h2>How It Works</h2>
              <p>
                The analysis pipeline combines forensic feature extraction with deep visual classification.
              </p>
            </div>
          </div>

          <div className="timeline">
            <div className="timelineItem">
              <h3>1. Upload</h3>
              <p>The user submits a suspicious image for evaluation.</p>
            </div>

            <div className="timelineItem">
              <h3>2. Feature Extraction</h3>
              <p>ELA and noise residual maps are generated to reveal hidden artifacts.</p>
            </div>

            <div className="timelineItem">
              <h3>3. Deep Inference</h3>
              <p>A 7-channel CBAM-ResNet50 predicts authenticity and manipulation type.</p>
            </div>

            <div className="timelineItem">
              <h3>4. Localization</h3>
              <p>Grad-CAM highlights the most suspicious region to support interpretation.</p>
            </div>
          </div>
        </section>

        <section className="panel notePanel">
          <h2>Important Note</h2>
          <p>
            This system is intended for academic and research use. Predictions and visual
            explanations should support forensic review, not replace expert judgment.
          </p>
        </section>
      </main>

      <footer className="footer">
        <div className="footerInner">
          <strong>ForenSight</strong>
          <p>
            AI-based digital image tampering detection using forensic feature fusion and visual localization.
          </p>
        </div>
      </footer>
    </div>
  );
}