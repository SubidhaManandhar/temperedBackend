function notFound(req, res, next) {
  res.status(404).json({ error: "Route not found" });
}

function errorHandler(err, req, res, next) {
  const message = err.message || "Server error";
  res.status(500).json({ error: message });
}

module.exports = { notFound, errorHandler };