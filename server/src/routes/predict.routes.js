const express = require("express");
const router = express.Router();

const upload = require("../middlewares/upload.middleware");
const { predictImage } = require("../controllers/predict.controller");

router.post("/predict", upload.single("image"), predictImage);

module.exports = router;