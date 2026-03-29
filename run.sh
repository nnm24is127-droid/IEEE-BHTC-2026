#!/bin/bash
# ──────────────────────────────────────────────
# ParkMod – run.sh
# Quick launcher for the ParkMod system
# ──────────────────────────────────────────────

echo "=============================================="
echo "  ParkMod: AI-Based Intelligent Parking"
echo "  Enforcement System"
echo "=============================================="

# Install dependencies
echo "[*] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Choose run mode:"
echo "  1) Streamlit Dashboard (Web UI)"
echo "  2) CLI – Demo mode"
echo "  3) CLI – Video file"
echo ""
read -p "Enter choice [1/2/3]: " choice

case $choice in
    1)
        echo "[*] Launching Streamlit dashboard..."
        streamlit run app.py --server.headless true
        ;;
    2)
        echo "[*] Running CLI in demo mode..."
        python -m src.main --demo --threshold 10
        ;;
    3)
        read -p "Enter video file path: " video_path
        echo "[*] Processing video: $video_path"
        python -m src.main --video "$video_path" --threshold 10
        ;;
    *)
        echo "[!] Invalid choice. Launching dashboard..."
        streamlit run app.py --server.headless true
        ;;
esac
