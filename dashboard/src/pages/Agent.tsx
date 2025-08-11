import React, { useState } from 'react';
import { Typography, Box, TextField, Button, CircularProgress, Paper } from '@mui/material';
import axios from 'axios';

const Agent = () => {
  const [goal, setGoal] = useState('Run prioritized tests for changed files');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleRunAgent = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/agent', {
        goal,
        context: {},
      });
      setResult(resp.data.result);
    } catch (e) {
      setResult({ error: 'Agent run failed.' });
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        AI Agent
      </Typography>
      <Typography variant="body1" sx={{ mb: 2 }}>
        Interact with the LangChain AI agent for automated testing tasks.
      </Typography>
      <TextField label="Goal" value={goal} onChange={e => setGoal(e.target.value)} fullWidth sx={{ mb: 2 }} />
      <Button variant="contained" onClick={handleRunAgent} disabled={loading}>Run Agent</Button>
      {loading && <CircularProgress sx={{ mt: 2 }} />}
      {result && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="body2">{typeof result === 'string' ? result : JSON.stringify(result)}</Typography>
        </Paper>
      )}
    </Box>
  );
};

export default Agent;