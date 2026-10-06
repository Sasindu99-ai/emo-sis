const { pool } = require('./connection');

// Patients
async function createPatient({ identifier, roomNumber, bedIdentifier, notes = null }) {
  const [result] = await pool.execute(
    `INSERT INTO patients (patient_identifier, room_number, bed_identifier, notes)
     VALUES (?, ?, ?, ?)`,
    [identifier, roomNumber, bedIdentifier, notes]
  );
  return { id: result.insertId, identifier, roomNumber, bedIdentifier };
}

async function getPatientById(id) {
  const [rows] = await pool.execute(
    `SELECT * FROM patients WHERE id = ?`,
    [id]
  );
  return rows[0] || null;
}

async function listActivePatients() {
  const [rows] = await pool.execute(
    `SELECT * FROM patients WHERE status = 'active' ORDER BY room_number, bed_identifier`
  );
  return rows;
}

// Monitoring Windows
async function createMonitoringWindow({ patientId, windowStart, windowEnd }) {
  const [result] = await pool.execute(
    `INSERT INTO monitoring_windows (patient_id, window_start, window_end)
     VALUES (?, ?, ?)`,
    [patientId, windowStart, windowEnd]
  );
  return { id: result.insertId, patientId, windowStart, windowEnd };
}

// Inferences
async function recordInference({ windowId, branch, scores, distressScore = null, details = null }) {
  const [result] = await pool.execute(
    `INSERT INTO inferences (window_id, branch, scores, distress_score, details)
     VALUES (?, ?, ?, ?, ?)`,
    [
      windowId,
      branch,
      JSON.stringify(scores),
      distressScore,
      details ? JSON.stringify(details) : null
    ]
  );
  return { id: result.insertId, windowId, branch };
}

async function getInferencesByWindow(windowId) {
  const [rows] = await pool.execute(
    `SELECT * FROM inferences WHERE window_id = ? ORDER BY created_at ASC`,
    [windowId]
  );
  return rows;
}

// Alerts
async function createAlert({ windowId, patientId, distressScore, severity = 'medium' }) {
  const [result] = await pool.execute(
    `INSERT INTO alerts (window_id, patient_id, distress_score, severity, status)
     VALUES (?, ?, ?, ?, 'active')`,
    [windowId, patientId, distressScore, severity]
  );
  return { id: result.insertId, windowId, patientId, distressScore, severity, status: 'active' };
}

async function acknowledgeAlert(alertId, acknowledgedBy, notes = null) {
  const [result] = await pool.execute(
    `UPDATE alerts
     SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = CURRENT_TIMESTAMP, resolution_notes = ?
     WHERE id = ?`,
    [acknowledgedBy, notes, alertId]
  );
  return result.affectedRows > 0;
}

async function listActiveAlerts() {
  const [rows] = await pool.execute(
    `SELECT a.*, p.patient_identifier, p.room_number, p.bed_identifier
     FROM alerts a
     JOIN patients p ON a.patient_id = p.id
     WHERE a.status = 'active'
     ORDER BY a.created_at DESC`
  );
  return rows;
}

module.exports = {
  createPatient,
  getPatientById,
  listActivePatients,
  createMonitoringWindow,
  recordInference,
  getInferencesByWindow,
  createAlert,
  acknowledgeAlert,
  listActiveAlerts
};
