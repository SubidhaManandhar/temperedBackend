const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");

const predictRoutes = require("./routes/predict.routes");
const { notFound, errorHandler } = require("./middlewares/error.middleware");

const app = express();

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "5mb" }));
app.use(morgan("dev"));

app.get("/health", (req, res) => {
  res.json({ ok: true, message: "Backend is running!" });
});

app.use("/api/predict", predictRoutes);

app.use(notFound);
app.use(errorHandler);

module.exports = app;