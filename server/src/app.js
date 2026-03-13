const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const helmet = require("helmet");
const path = require("path");

const predictRoutes = require("./routes/predict.routes");

const app = express();

app.use(cors());
app.use(helmet());
app.use(morgan("dev"));
app.use(express.json());

app.use("/api", predictRoutes);
app.use("/static", express.static(path.join(process.cwd(), "ml_service", "outputs")));
app.use(express.static(path.join(__dirname, "../public")));

app.get("/health", (req, res) => {
  res.json({ ok: true, service: "node-backend" });
});

app.use((req, res, next) => {
  if (
    req.path.startsWith("/api") ||
    req.path.startsWith("/static") ||
    req.path === "/health"
  ) {
    return next();
  }

  res.sendFile(path.join(__dirname, "../public", "index.html"));
});

module.exports = app;