import React from 'react';
import axios from 'axios';
import { useQuery } from '@tanstack/react-query';
import { Grid, Card, CardContent, Typography, Box, LinearProgress, Chip } from '@mui/material';
import { BugReport as BugIcon, Code as CodeIcon, PlayArrow as PlayIcon, TrendingUp as TrendingIcon } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Simple types for chart and analytics
type Run = { result?: { risk?: number } };

type Analytics = {
  pass?: number;
  fail?: number;
  defect_predictions?: any[];
};

type StatCardProps = {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
};

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, subtitle }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ color, mr: 2 }}>{icon}</Box>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" sx={{ color, mb: 1 }}>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="body2" color="text.secondary">
          {subtitle}
        </Typography>
      )}
    </CardContent>
  </Card>
);

const Dashboard: React.FC = () => {
  const { data } = useQuery<{ analytics?: Analytics; runs?: Run[] }>({
    queryKey: ['analytics'],
    queryFn: async () => {
      const resp = await axios.get('/analytics');
      return resp.data as { analytics?: Analytics; runs?: Run[] };
    },
    refetchInterval: 5000,
  });

  const analytics: Analytics = data?.analytics || {};
  const runs: Run[] = data?.runs || [];

  const stats = {
    totalTests: runs.length,
    passingTests: analytics.pass || 0,
    failingTests: analytics.fail || 0,
    defectPredictions: (analytics.defect_predictions || []).length,
    testCoverage: Math.round(((analytics.pass || 0) / Math.max(runs.length, 1)) * 100),
  };

  const chartData = runs.slice(-7).map((run, idx) => ({
    day: `Run ${Math.max(runs.length - 7, 0) + idx + 1}`,
    tests: 1,
    defects: run.result?.risk ?? 0,
  }));

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard Overview
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Tests"
            value={stats.totalTests}
            icon={<CodeIcon />}
            color="#1976d2"
            subtitle="Automated test cases"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Passing Tests"
            value={stats.passingTests}
            icon={<PlayIcon />}
            color="#2e7d32"
            subtitle={`${Math.round((stats.passingTests / Math.max(stats.totalTests, 1)) * 100)}% success rate`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Failing Tests"
            value={stats.failingTests}
            icon={<BugIcon />}
            color="#d32f2f"
            subtitle="Requires attention"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Defect Predictions"
            value={stats.defectPredictions}
            icon={<TrendingIcon />}
            color="#ed6c02"
            subtitle="AI-predicted risks"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Test Execution Trends
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="tests" stroke="#1976d2" strokeWidth={2} />
                  <Line type="monotone" dataKey="defects" stroke="#d32f2f" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Test Coverage
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4" component="div" sx={{ mr: 2 }}>
                  {stats.testCoverage}%
                </Typography>
                <Chip
                  label={stats.testCoverage >= 80 ? 'Good' : stats.testCoverage >= 60 ? 'Fair' : 'Poor'}
                  color={stats.testCoverage >= 80 ? 'success' : stats.testCoverage >= 60 ? 'warning' : 'error'}
                  size="small"
                />
              </Box>
              <LinearProgress variant="determinate" value={stats.testCoverage} sx={{ height: 8, borderRadius: 4 }} />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Target: 80% coverage
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
 