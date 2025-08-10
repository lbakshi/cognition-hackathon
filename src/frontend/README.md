# Frontend

Minimal static site served by Express for Railway deployments.

## Development

```bash
npm install
npm start
```

The server serves files from `public/` and listens on the `PORT` environment variable (defaults to `3000`).

## Deployment

Use the default Node.js environment on Railway with the start command:

```
npm start
```

Set the `API_BASE_URL` environment variable to the URL of your FastAPI
backend (e.g. `https://your-server.up.railway.app`) so the frontend can
communicate with it.
