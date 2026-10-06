const express = require('express');
const router = express.Router();
const db = require('../db/queries');

router.get('/', async (req, res) => {
  try {
    const alerts = await db.listActiveAlerts();
    res.json(alerts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/:id/acknowledge', async (req, res) => {
  try {
    const alertId = parseInt(req.params.id, 10);
    const { acknowledgedBy, notes } = req.body;
    if (!acknowledgedBy) {
      return res.status(400).json({ error: 'acknowledgedBy is required' });
    }
    const success = await db.acknowledgeAlert(alertId, acknowledgedBy, notes);
    if (!success) {
      return res.status(404).json({ error: 'Alert not found or already acknowledged' });
    }
    res.json({ message: 'Alert acknowledged successfully', alertId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
