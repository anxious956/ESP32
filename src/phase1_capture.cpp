// Phase 1 — labeled audio window capture
//
// Captures fixed-length audio windows on command and streams them over serial
// so a Python tool can save labeled training data (clap / whistle / snap /
// silence). Run phase0_miccheck FIRST to confirm the mic works and tune the pot.
//
// Sampling: a dedicated micros()-paced busy-wait loop. During a capture the MCU
// does nothing else, so pacing jitter is tiny and the rate is deterministic —
// this is NOT the same as scattering analogRead() through loop(). (Phase 3's
// real-time continuous path will move to DMA/I2S; for discrete window capture a
// tight paced loop is the simplest robust choice and is version-independent.)
//
// Wiring (same as Phase 0): mic OUT->GPIO34, VCC->3V3, GND->GND.
// Serial: 921600 baud (fast bulk transfer of 8000-sample windows).
//
// Protocol:
//   host sends  'c'  -> board captures one window, then streams:
//       BEGIN <N> <SAMPLE_RATE>
//       v0,v1,...,v(N-1)
//       END

#include <Arduino.h>

constexpr int      MIC_PIN     = 34;                       // ADC1_CH6, input-only
constexpr uint32_t SAMPLE_RATE = 16000;                    // Hz (matches ml/features.py)
constexpr uint32_t WINDOW_MS   = 500;                      // per window
constexpr uint32_t N           = SAMPLE_RATE * WINDOW_MS / 1000;  // 8000 samples

static uint16_t buf[N];

static void captureWindow() {
  const uint32_t t0 = micros();
  for (uint32_t i = 0; i < N; ++i) {
    // absolute schedule (no cumulative drift): sample i is due at i/SR seconds
    const uint32_t target = t0 + (uint32_t)((uint64_t)i * 1000000ULL / SAMPLE_RATE);
    while ((int32_t)(micros() - target) < 0) { /* pace to the sample clock */ }
    buf[i] = analogRead(MIC_PIN);
  }
}

static void streamWindow() {
  Serial.printf("BEGIN %lu %lu\n", (unsigned long)N, (unsigned long)SAMPLE_RATE);
  for (uint32_t i = 0; i < N; ++i) {
    Serial.print(buf[i]);
    Serial.print((i + 1 < N) ? ',' : '\n');
  }
  Serial.println("END");
}

void setup() {
  Serial.begin(921600);
  delay(300);
  analogReadResolution(12);                          // 0..4095
  analogSetPinAttenuation(MIC_PIN, ADC_11db);        // full ~0..3.3 V span
  Serial.printf("# Phase 1 capture ready — %lu samples @ %lu Hz (%lu ms).\n",
                (unsigned long)N, (unsigned long)SAMPLE_RATE, (unsigned long)WINDOW_MS);
  Serial.println("# send 'c' to capture one window");
}

void loop() {
  if (Serial.available()) {
    const int c = Serial.read();
    if (c == 'c' || c == 'C') {
      captureWindow();
      streamWindow();
    }
    // any other byte (e.g. the newline after 'c') is ignored
  }
}
