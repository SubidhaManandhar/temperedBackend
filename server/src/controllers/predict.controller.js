const fs = require("fs");
const FormData = require("form-data");
const client = require("../utils/httpClient.js");

async function predictImage(req, res, next) {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Image file is required." });
    }

    const mlUrl = process.env.ML_SERVICE_URL;

    // If Python service is OFF, return dummy response
    if (!mlUrl) {
      return res.json({
        filename: req.file.filename,
        label: "unknown",
        confidence: 0,
        note: "ML_SERVICE_URL not set",
      });
    }

    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    const mlRes = await client.post(`${mlUrl}/predict`, form, {
      headers: form.getHeaders(),
    });

    return res.json({
      filename: req.file.filename,
      ...mlRes.data,
    });
  } catch (err) {
    next(err);
  }
}

module.exports = { predictImage };
