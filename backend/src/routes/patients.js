const express = require('express');
const router = express.Router();
const db = require('../db/queries');

router.get('/', async (req, res) => {
  try {
    const patients = await db.listActivePatients();
    res.json(patients);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/', async (req, res) => {
  try {
    const { identifier, roomNumber, bedIdentifier, notes } = req.body;
    if (!identifier || !roomNumber || !bedIdentifier) {
      return res.status(400).json({ error: 'identifier, roomNumber, and bedIdentifier are required' });
    }
    const created = await db.createPatient({ identifier, roomNumber, bedIdentifier, notes });
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
