const axios = require("axios");

const client = axios.create({
  timeout: 120000
});

module.exports = client;