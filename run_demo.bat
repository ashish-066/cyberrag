@echo off
rem Launch CyberRAG Analyst Workbench with Demo Overlay enabled
echo ==========================================================
echo Starting CyberRAG Analyst Workbench (http://127.0.0.1:8080)
echo Mode: Demo Workbench with Simulated Temporal Overlay Enabled
echo ==========================================================

set CTI_LOAD_DEMO_OVERLAY=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

python demo/server.py
