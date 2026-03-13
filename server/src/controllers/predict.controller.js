const fs = require("fs");
const FormData = require("form-data");
const axios = require("axios");

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

const predictImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded." });
    }

    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    const response = await axios.post(`${ML_SERVICE_URL}/predict`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    return res.json({
      filename: req.file.filename,
      ...response.data,
    });
  } catch (error) {
    const message =
      error?.response?.data?.detail ||
      error?.response?.data?.error ||
      error.message ||
      "Prediction failed";

    return res.status(500).json({ error: message });
  }
};

module.exports = { predictImage };