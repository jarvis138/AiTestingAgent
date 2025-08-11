import React, { useState } from 'react';
import { Typography, Box, TextField, Button, CircularProgress, Paper } from '@mui/material';
import axios from 'axios';

const TestExecution = () => {
  const [testFile, setTestFile] = useState('playwright/tests/example.spec.ts');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleRunTest = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/run_tests', {
        tests: [testFile],
      });
      setResult(resp.data);
    } catch (e) {
      setResult({ error: 'Test execution failed.' });
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Test Execution
      </Typography>
      <Typography variant="body1" sx={{ mb: 2 }}>
        Monitor and manage test execution in sandboxed environments.
      </Typography>
      <TextField label="Test File" value={testFile} onChange={e => setTestFile(e.target.value)} sx={{ mr: 2 }} />
      <Button variant="contained" onClick={handleRunTest} disabled={loading}>Run Test</Button>
      {loading && <CircularProgress sx={{ mt: 2 }} />}
      {result && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="body2">{result.stdout ? result.stdout : JSON.stringify(result)}</Typography>
        </Paper>
      )}
    </Box>
  );
};

export default TestExecution;