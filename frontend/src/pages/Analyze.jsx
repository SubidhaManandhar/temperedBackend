import { useEffect, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import ReportCard from "../components/ReportCard";
import EvidenceGrid from "../components/EvidenceGrid";

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

export default function Analyze() {
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
      setTimeout(() => {
        document.getElementById("resultsSection")?.scrollIntoView({ behavior: "smooth" });
      }, 100);
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
      <Navbar />

      <section className="analyzeHero">
        <div className="heroInner">
          <p className="eyebrow">Interactive Analysis</p>
          <h1>Submit Image for Forensic Analysis</h1>
          <p className="heroText">
            Upload a suspicious image and review authenticity, forensic maps,
            localization evidence, and the final analysis report.
          </p>
        </div>
      </section>

      <main className="pageWrap">
        <section className="panel" id="analyze">
          <div className="sectionTop">
            <div>
              <h2>Upload Evidence</h2>
              <p>
                Supported formats: JPG, JPEG, PNG
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

        <div id="resultsSection">
          {data ? (
            <>
              <ReportCard data={data} />
              <EvidenceGrid data={data} />
            </>
          ) : (
            <section className="panel">
              <div className="placeholderBox">
                Run an analysis to generate the forensic report and evidence visualizations.
              </div>
            </section>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}