import React, { useState } from 'react';
import { Typography, Box, Button, CircularProgress, Paper } from '@mui/material';
import axios from 'axios';

const TestGeneration = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleGenerateTest = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/generate_test', {
        source: { type: 'demo' },
      });
      setResult(resp.data);
    } catch (e) {
      setResult({ error: 'Test generation failed.' });
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Test Generation
      </Typography>
      <Typography variant="body1" sx={{ mb: 2 }}>
        AI-powered test code generation and customization.
      </Typography>
      <Button variant="contained" onClick={handleGenerateTest} disabled={loading}>Generate Sample Test</Button>
      {loading && <CircularProgress sx={{ mt: 2 }} />}
      {result && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="body2">{result.code ? result.code : JSON.stringify(result)}</Typography>
        </Paper>
      )}
    </Box>
  );
};

export default TestGeneration;