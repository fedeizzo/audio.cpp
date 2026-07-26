#!/usr/bin/env bash
set -euo pipefail
# Run audiocpp_cli under rocprof to capture GPU kernel timing on gfx1151.

TEXT_MODE="${1:-short}"
case "$TEXT_MODE" in
    short)  TEXT="Ciao mondo, questo e un test di sintesi vocale." ;;
    medium) TEXT="L intelligenza artificiale sta trasformando il modo in cui interagiamo con la tecnologia. I modelli di sintesi vocale neurale possono ora generare parlato naturale con incredibile fedelta." ;;
    long)   TEXT="La storia dell esplorazione spaziale rappresenta una delle avventure piu affascinanti e ambiziose dell umanita. Dalla conquista della Luna con le missioni Apollo negli anni sessanta, fino all invio di rover robotici sulla superficie di Marte, l uomo ha sempre cercato di spingersi oltre i confini del conosciuto." ;;
esac

HIP_CLI="./build/linux-hip-release/bin/audiocpp_cli"
MODEL="/persist/models/audio.cpp/Qwen3-TTS-12Hz-0.6B-Base/model_q8_0.gguf"
VOICE_REF="/persist/models/audio.cpp/clear-italian-voice.wav"
REF_TEXT="Questo racconto e cresciuto nel corso della narrazione fino a diventare una storia della Grande Guerra dell Anello, e ha in."
OUT_DIR="rocprof_output"
mkdir -p "$OUT_DIR"

echo "=== HIP rocprof profile (mode=$TEXT_MODE) ==="

nix develop .#rocm -c rocprof --hip-trace --stats -d "$OUT_DIR" \
  "$HIP_CLI" \
  --task tts --family qwen3_tts \
  --model "$MODEL" \
  --backend hip \
  --voice-ref "$VOICE_REF" \
  --reference-text "$REF_TEXT" \
  --text "$TEXT" \
  --seed 42 --log \
  --out "$OUT_DIR/output.wav" \
  2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "=== Top 20 GPU kernels by time ==="
if [ -f "$OUT_DIR/results.stats.csv" ]; then
    head -1 "$OUT_DIR/results.stats.csv"
    tail -n +2 "$OUT_DIR/results.stats.csv" | sort -t',' -k2 -rn | head -20
else
    echo "No results.stats.csv found in $OUT_DIR/"
    ls -la "$OUT_DIR/"
fi

echo ""
echo "=== Per-step timing (from run log) ==="
grep "TIMING.*talker.cached_step\|TIMING.*talker.code_predictor\|TIMING.*speech_decoder\|TIMING.*session.wall" "$OUT_DIR/run.log" | head -20