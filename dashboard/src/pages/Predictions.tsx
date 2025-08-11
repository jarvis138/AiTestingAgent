import React, { useState } from 'react';
import { Typography, Box, TextField, Button, CircularProgress, Paper } from '@mui/material';
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

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Defect Predictions
      </Typography>
      <Box sx={{ mb: 2 }}>
        <TextField label="File path" value={file} onChange={e => setFile(e.target.value)} sx={{ mr: 2 }} />
        <Button variant="contained" onClick={handlePredict} disabled={loading}>Predict</Button>
      </Box>
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