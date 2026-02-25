const express = require("express");
const { upload } = require("../middlewares/upload.middleware");
const { predictImage } = require("../controllers/predict.controller");

const router = express.Router();

// POST /api/predict (form-data with key "image")
router.post("/", upload.single("image"), predictImage);

module.exports = router;