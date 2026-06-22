// Phase 0 — Mic check
// Confirms the ESP32 ADC actually "sees" sound on the KY-038 analog out (AO),
// before we commit to the full sampling + feature + ML pipeline.
//
// Wiring:
//   Mic AO  -> GPIO34  (ADC1_CH6, input-only, safe with WiFi)
//   Mic VCC -> 3V3     (NOT 5V — AO can exceed 3.3 V and damage the ADC)
//   Mic GND -> GND
//
// Output: one CSV line per ~100 ms -> min,max,mean,rms_ac,p2p
//   rms_ac = RMS after removing the DC bias = the "loudness" signal we care about.
//   Read it with tools/serial_logger.py for a live bar + clap pass/fail.

#include <Arduino.h>
#include <math.h>

constexpr int      MIC_AO_PIN = 34;     // ADC1_CH6 (input-only)
constexpr int      WINDOW     = 800;    // samples per report window
constexpr uint32_t REPORT_MS  = 100;    // report cadence (ms)

static uint16_t buf[WINDOW];

void setup() {
  Serial.begin(115200);
  delay(300);
  analogReadResolution(12);                        // 0..4095
  analogSetPinAttenuation(MIC_AO_PIN, ADC_11db);   // full ~0..3.3 V input span
  Serial.println(F("# Phase 0 mic check — GPIO34, ADC1, 12-bit, 11dB"));
  Serial.println(F("# CSV: min,max,mean,rms_ac,p2p"));
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last < REPORT_MS) return;
  last = millis();

  // --- pass 1: collect window, track min/max/mean ---
  uint16_t vmin = 4095, vmax = 0;
  double   sum  = 0;
  for (int i = 0; i < WINDOW; ++i) {
    uint16_t v = analogRead(MIC_AO_PIN);
    buf[i] = v;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
    sum += v;
  }
  const double mean = sum / WINDOW;

  // --- pass 2: AC RMS (DC bias removed) ---
  double sq = 0;
  for (int i = 0; i < WINDOW; ++i) {
    const double d = (double)buf[i] - mean;
    sq += d * d;
  }
  const double   rms = sqrt(sq / WINDOW);
  const uint16_t p2p = vmax - vmin;

  Serial.printf("%u,%u,%.1f,%.1f,%u\n", vmin, vmax, mean, rms, p2p);
}
