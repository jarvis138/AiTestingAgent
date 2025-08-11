import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Paper, Stack } from '@mui/material';
import axios from 'axios';

const Jira: React.FC = () => {
  const [jql, setJql] = useState('project = DEMO ORDER BY created DESC');
  const [issues, setIssues] = useState<any[]>([]);
  const [issueKey, setIssueKey] = useState('');
  const [issue, setIssue] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [genResult, setGenResult] = useState<any>(null);

  const search = async () => {
    setLoading(true);
    try {
      const resp = await axios.post('/jira/search', { jql, max_results: 20 });
      setIssues(resp.data.issues || []);
    } catch (e) { setIssues([]); }
    setLoading(false);
  };

  const loadIssue = async () => {
    setLoading(true);
    try {
      const resp = await axios.get(`/jira/issue/${issueKey}`);
      setIssue(resp.data);
    } catch (e) { setIssue({ error: 'Failed to load issue' }); }
    setLoading(false);
  };

  const generateFromIssue = async () => {
    if (!issue) return;
    setLoading(true);
    try {
      const source = { type: 'jira', issue: { key: issue.key, summary: issue.summary, description: issue.description } };
      const resp = await axios.post('/generate_test', { source });
      setGenResult(resp.data);
    } catch (e) { setGenResult({ error: 'Generation failed' }); }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>JIRA Integration</Typography>
      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField fullWidth label="JQL" value={jql} onChange={(e) => setJql(e.target.value)} />
        <Button variant="contained" onClick={search} disabled={loading}>Search</Button>
      </Stack>
      {issues.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1">Issues</Typography>
          {issues.map((it) => (
            <Box key={it.key} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.5 }}>
              <span>{it.key} — {it.summary} ({it.status})</span>
              <Button size="small" onClick={() => { setIssueKey(it.key); setIssue(null); }}>Select</Button>
            </Box>
          ))}
        </Paper>
      )}

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <TextField label="Issue Key" value={issueKey} onChange={(e) => setIssueKey(e.target.value)} sx={{ minWidth: 240 }} />
        <Button variant="outlined" onClick={loadIssue} disabled={loading}>Load Issue</Button>
        <Button variant="contained" onClick={generateFromIssue} disabled={loading || !issue}>Generate Test</Button>
      </Stack>

      {issue && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1">Issue Details</Typography>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(issue, null, 2)}</pre>
        </Paper>
      )}

      {genResult && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="subtitle1">Generation Result</Typography>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{typeof genResult === 'string' ? genResult : JSON.stringify(genResult, null, 2)}</pre>
        </Paper>
      )}
    </Box>
  );
};

export default Jira;
