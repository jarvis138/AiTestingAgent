import React, { useState } from 'react';
import { Typography, Box, TextField, Button, CircularProgress, Paper, Stack } from '@mui/material';
import axios from 'axios';

const Predictions = () => {
  const [file, setFile] = useState('src/app/login.py');
  const [features, setFeatures] = useState({ loc: 400, complexity: 12, churn: 5, num_devs: 2 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/predict', {
        repo: 'demo',
        file,
        features,
      });
      setResult(resp.data);
    } catch (e) {
      setResult({ error: 'Prediction failed.' });
    }
    setLoading(false);
  };

  const handleExplain = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/explain', {
        repo: 'demo',
        file,
        features,
      });
      setResult(resp.data);
    } catch (e) {
      setResult({ error: 'Explain failed (ensure model with feature_columns is trained).' });
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Defect Predictions
      </Typography>
      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField label="File path" value={file} onChange={e => setFile(e.target.value)} />
        <Button variant="contained" onClick={handlePredict} disabled={loading}>Predict</Button>
        <Button variant="outlined" onClick={handleExplain} disabled={loading}>Explain</Button>
      </Stack>
      {loading && <CircularProgress />}
      {result && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="body1">Risk: {result.risk}</Typography>
          <Typography variant="body2">Explanation: {JSON.stringify(result.explanation)}</Typography>
        </Paper>
      )}
    </Box>
  );
};

export default Predictions;