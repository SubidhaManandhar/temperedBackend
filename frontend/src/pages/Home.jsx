import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

export default function Home() {
  return (
    <div className="siteShell">
      <Navbar />

      <section className="hero heroAlt">
        <div className="heroInner">
          <p className="eyebrow">AI-Based Image Tampering Detection</p>
          <h1>Understand image authenticity through forensic evidence.</h1>
          <p className="heroText">
            ForenSight combines Error Level Analysis, noise residual mapping,
            deep classification, and GradCAM localization to detect manipulated
            images and explain why they were flagged.
          </p>

          <div className="heroActions">
            <Link to="/analyze" className="primaryLinkBtn">
              Analyze Image
            </Link>
            <a href="#features" className="secondaryLinkBtn">
              Explore Features
            </a>
          </div>
        </div>
      </section>

      <main className="pageWrap homeWrap">
        <section className="panel">
          <div className="sectionTop">
            <div>
              <h2>About the Project</h2>
              <p>
                This system is designed to help identify manipulated images by
                combining forensic feature extraction with deep neural
                classification and visual localization.
              </p>
            </div>
          </div>

          <div className="infoGrid">
            <div className="infoCard">
              <h3>Deep Classification</h3>
              <p>
                A ResNet-based model processes RGB, ELA, and noise inputs to
                determine whether an image is authentic or tampered.
              </p>
            </div>

            <div className="infoCard">
              <h3>Explainable Outputs</h3>
              <p>
                GradCAM and binary masks highlight suspicious regions, making the
                result easier to inspect and interpret.
              </p>
            </div>

            <div className="infoCard">
              <h3>Forensic Evidence</h3>
              <p>
                The platform displays ELA, noise maps, heatmaps, and overlays to
                support the final decision visually.
              </p>
            </div>
          </div>
        </section>

        <section className="panel" id="features">
          <div className="sectionTop">
            <div>
              <h2>Core Features</h2>
              <p>These outputs are available through the analysis workflow.</p>
            </div>
          </div>

          <div className="featureGrid">
            <div className="featureCard">
              <h3>1. GradCAM Heatmap</h3>
              <p>Highlights the region most responsible for the model prediction.</p>
            </div>
            <div className="featureCard">
              <h3>2. ELA Visualization</h3>
              <p>Reveals compression inconsistencies often associated with edits.</p>
            </div>
            <div className="featureCard">
              <h3>3. Noise Residual Map</h3>
              <p>Shows abnormal sensor-noise and residual patterns.</p>
            </div>
            <div className="featureCard">
              <h3>4. Binary Suspicion Mask</h3>
              <p>Extracts and isolates the most suspicious region.</p>
            </div>
            <div className="featureCard">
              <h3>5. Annotated Overlay</h3>
              <p>Places localization evidence directly on top of the original image.</p>
            </div>
            <div className="featureCard">
              <h3>6. Full Forensic Report</h3>
              <p>Summarizes authenticity, class prediction, confidence, and localization.</p>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="sectionTop">
            <div>
              <h2>Example Output</h2>
              <p>This example shows the kind of evidence the user will receive after analysis.</p>
            </div>
          </div>

          <div className="exampleGrid">
            <div className="exampleBox">Original Image</div>
            <div className="exampleBox">ELA Map</div>
            <div className="exampleBox">Noise Residual</div>
            <div className="exampleBox">GradCAM Heatmap</div>
            <div className="exampleBox">Binary Mask</div>
            <div className="exampleBox">Annotated Overlay</div>
          </div>
        </section>

        <section className="panel" id="workflow">
          <div className="sectionTop">
            <div>
              <h2>Workflow</h2>
              <p>The analysis process is transparent and evidence-driven.</p>
            </div>
          </div>

          <div className="timeline">
            <div className="timelineItem">
              <h3>1. Upload Image</h3>
              <p>The user submits a suspicious image for analysis.</p>
            </div>
            <div className="timelineItem">
              <h3>2. Generate Maps</h3>
              <p>ELA and noise-residual maps are computed from the uploaded image.</p>
            </div>
            <div className="timelineItem">
              <h3>3. Predict Class</h3>
              <p>The deep model predicts authenticity and, if tampered, the manipulation type.</p>
            </div>
            <div className="timelineItem">
              <h3>4. Localize Region</h3>
              <p>GradCAM and binary masking localize the suspicious area visually.</p>
            </div>
          </div>
        </section>

        <section className="panel notePanel">
          <h2>Important Note</h2>
          <p>
            This tool is intended for academic and research use. It should support
            forensic review and interpretation, not replace expert judgment.
          </p>
        </section>
      </main>

      <Footer />
    </div>
  );
}