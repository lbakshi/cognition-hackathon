const express = require('express');
const path = require('path');

const app = express();

// Serve a small JS file that exposes the backend API URL so the
// frontend can talk to the separately hosted FastAPI server.
app.get('/config.js', (_req, res) => {
  const apiBase = process.env.API_BASE_URL || '';
  res.type('application/javascript');
  res.send(`window.API_BASE_URL = '${apiBase}';`);
});

// Static assets for the app live in /public.
const publicPath = path.join(__dirname, 'public');
app.use(express.static(publicPath));

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Frontend server running on port ${port}`);
});
