@echo off
cd /d "C:\VoiceChanger-RVC"
set "RVC_CUDA_GRAPH=0"
start "RVC Voice Changer" "C:\VoiceChanger-RVC\.venv\Scripts\pythonw.exe" "C:\VoiceChanger-RVC\realtime_gui.py"
