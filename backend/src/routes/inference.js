const express = require('express');
const router = express.Router();
const db = require('../db/queries');
const { broadcastAlert } = require('../sockets/alertSocket');

const PYTHON_FUSION_URL = process.env.PYTHON_FUSION_URL || 'http://localhost:8000';

router.post('/record', async (req, res) => {
  try {
    const { patientId, windowStart, windowEnd, facial, audio, text, fused } = req.body;

    if (!patientId || !fused) {
      return res.status(400).json({ error: 'patientId and fused results are required' });
    }

    // 1. Create monitoring window
    const window = await db.createMonitoringWindow({
      patientId,
      windowStart: windowStart ? new Date(windowStart) : new Date(),
      windowEnd: windowEnd ? new Date(windowEnd) : new Date()
    });

    // 2. Persist branch inferences
    if (facial) {
      await db.recordInference({
        windowId: window.id,
        branch: 'facial',
        scores: facial,
        distressScore: (facial.pain || 0) + (facial.panic || 0)
      });
    }
    if (audio) {
      await db.recordInference({
        windowId: window.id,
        branch: 'audio',
        scores: audio,
        distressScore: (audio.fear || 0) + (audio.sad || 0) + (audio.distress || 0)
      });
    }
    if (text) {
      await db.recordInference({
        windowId: window.id,
        branch: 'text',
        scores: { transcript: text.transcript },
        distressScore: text.distress_score,
        details: text.keywords
      });
    }
    if (fused) {
      await db.recordInference({
        windowId: window.id,
        branch: 'fused',
        scores: fused,
        distressScore: fused.distress_score
      });
    }

    // 3. Trigger alert if distress crosses threshold
    let alertRecord = null;
    if (fused.alert) {
      const severity = fused.distress_score >= 0.8 ? 'critical' : (fused.distress_score >= 0.6 ? 'high' : 'medium');
      alertRecord = await db.createAlert({
        windowId: window.id,
        patientId,
        distressScore: fused.distress_score,
        severity
      });

      // Broadcast real-time alert to nurse station dashboard via WebSocket
      const patient = await db.getPatientById(patientId);
      broadcastAlert({
        alertId: alertRecord.id,
        patientId,
        patientIdentifier: patient ? patient.patient_identifier : `Patient #${patientId}`,
        roomNumber: patient ? patient.room_number : 'Unknown',
        bedIdentifier: patient ? patient.bed_identifier : 'Unknown',
        distressScore: fused.distress_score,
        severity,
        windowId: window.id,
        timestamp: new Date().toISOString(),
        details: { facial, audio, text }
      });
    }

    res.status(201).json({
      windowId: window.id,
      patientId,
      alertGenerated: !!fused.alert,
      alertId: alertRecord ? alertRecord.id : null
    });
  } catch (err) {
    console.error('Error recording inference:', err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
