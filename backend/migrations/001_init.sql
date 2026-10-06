-- Multimodal Patient Distress Monitoring Schema
-- Migration 001: Initial Schema

CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_identifier VARCHAR(64) NOT NULL UNIQUE,
    room_number VARCHAR(32) NOT NULL,
    bed_identifier VARCHAR(32) NOT NULL,
    notes TEXT NULL,
    status ENUM('active', 'discharged', 'transferred') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_room_bed (room_number, bed_identifier),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS monitoring_windows (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_monitoring_windows_patient
        FOREIGN KEY (patient_id) REFERENCES patients(id)
        ON DELETE CASCADE,
    INDEX idx_patient_time (patient_id, window_start, window_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS inferences (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    window_id BIGINT NOT NULL,
    branch ENUM('facial', 'audio', 'text', 'fused') NOT NULL,
    scores JSON NOT NULL,
    distress_score FLOAT NULL,
    details JSON NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_inferences_window
        FOREIGN KEY (window_id) REFERENCES monitoring_windows(id)
        ON DELETE CASCADE,
    INDEX idx_window_branch (window_id, branch),
    INDEX idx_distress (distress_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    window_id BIGINT NOT NULL,
    patient_id INT NOT NULL,
    distress_score FLOAT NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    status ENUM('active', 'acknowledged', 'dismissed', 'resolved') DEFAULT 'active',
    acknowledged_by VARCHAR(128) NULL,
    acknowledged_at TIMESTAMP NULL,
    resolution_notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_alerts_window
        FOREIGN KEY (window_id) REFERENCES monitoring_windows(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_alerts_patient
        FOREIGN KEY (patient_id) REFERENCES patients(id)
        ON DELETE CASCADE,
    INDEX idx_patient_status (patient_id, status),
    INDEX idx_created (created_at),
    INDEX idx_severity (severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
