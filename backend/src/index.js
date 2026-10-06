const http = require('http');
const express = require('express');
const cors = require('cors');
require('dotenv').config();

const { testConnection } = require('./db/connection');
const { initSocket } = require('./sockets/alertSocket');
const patientsRouter = require('./routes/patients');
const alertsRouter = require('./routes/alerts');
const inferenceRouter = require('./routes/inference');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());

// Initialize WebSocket for real-time alerting
initSocket(server);

// API Routes
app.use('/api/patients', patientsRouter);
app.use('/api/alerts', alertsRouter);
app.use('/api/inference', inferenceRouter);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'emo-sis-backend', timestamp: new Date().toISOString() });
});

const PORT = parseInt(process.env.PORT || '5000', 10);

async function startServer() {
  await testConnection();
  server.listen(PORT, () => {
    console.log(`[emo-sis backend] Server listening on port ${PORT}`);
  });
}

if (require.main === module) {
  startServer();
}

module.exports = { app, server };
