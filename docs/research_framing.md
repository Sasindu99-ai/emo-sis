# Research Framing: Multimodal Patient Distress Monitoring

## 1. Problem Statement
In inpatient hospital wards and psychiatric observation units, scheduled nurse rounds typically occur every 30 to 120 minutes. Patients who are non-verbal, sedated, confused, post-operative, or experiencing acute psychological panic frequently experience severe distress in the gaps between scheduled rounds.

Because traditional telemetry focuses predominantly on hemodynamic vitals (ECG, SpO2, blood pressure) rather than psychological/behavioral state, non-verbal indicators of acute distress (facial grimacing, quiet moaning, agitation, crying) frequently go unaddressed until the next scheduled physical check-in.

## 2. Sensor-Fusion Formulation
We formulate patient distress detection as a late-fusion classification and triage problem over synchronized temporal windows (e.g. 3–5 seconds):

1. **Visual Signal ($S_v$)**: Video stream processed with face detection and lightweight convolutional neural network (MobileNetV2) mapping facial expressions into probabilities of pain, panic, agitation, or neutral rest.
2. **Acoustic Signal ($S_a$)**: Microphone stream filtered via spectral gating (`noisereduce`) to remove ward background fans/monitors, mapped via 2D CNN over log-mel spectrograms into vocal distress probabilities (fear/panic, sadness/crying, anger/agitation).
3. **Linguistic Signal ($S_t$)**: Speech transcription (Whisper / Google Speech Recognition) parsed for emergency keywords, distress phrases, and localized emergency terms (English, Sinhala, Tamil).

The signals are combined via a calibrated late-fusion engine:
$$\text{Distress Score} = f(S_v, S_a, S_t)$$

## 3. Clinical Triage Aid vs. Diagnostic Tool
- **Alert, Don't Diagnose**: The pipeline outputs a real-time distress score and alert trigger rather than clinical diagnoses.
- **Asymmetric Cost Matrix**: A false negative (failing to alert a patient in agony or panic) has severe consequences compared to an occasional false positive check. The system is calibrated with an operating threshold favoring recall on distress states.
