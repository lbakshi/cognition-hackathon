const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from server!' });
});

app.post('/api/plan', (req, res) => {
  const { prompt } = req.body || {};
  res.json({
    prompt,
    experimentSpec: '{"model": "Placeholder spec"}'
  });
});

app.post('/api/codegen', (req, res) => {
  res.json({
    files: {
      'train.py': '# train model\n',
      'eval.py': '# evaluate model\n',
      'model.py': '# model definition\n'
    }
  });
});

app.post('/api/execute', (req, res) => {
  res.json({
    metrics: { accuracy: 0.0 },
    plots: []
  });
});

app.post('/api/report', (req, res) => {
  res.json({
    summary: 'Placeholder summary.'
  });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
