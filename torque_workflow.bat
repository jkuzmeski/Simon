@echo off
REM Torque Workflow Commands for Isaac Lab
REM This script provides easy access to the torque-based motion imitation workflow

setlocal enabledelayedexpansion

REM Set the Isaac Lab path - MODIFY THIS TO YOUR ISAAC LAB INSTALLATION
set ISAAC_PATH=C:\isaac-sim
REM Alternative common paths:
REM set ISAAC_PATH=C:\Users\%USERNAME%\.local\share\ov\pkg\isaac-sim-2023.1.1
REM set ISAAC_PATH=C:\isaacsim

echo =========================================
echo Isaac Lab Torque Workflow Commands
echo =========================================

if "%1"=="" goto show_usage
if "%1"=="help" goto show_usage
if "%1"=="list" goto list_files
if "%1"=="collect" goto collect_torques
if "%1"=="convert" goto convert_csv
if "%1"=="train" goto train_amp
if "%1"=="demo" goto run_demo

:show_usage
echo.
echo Available commands:
echo   torque_workflow.bat help       - Show this help
echo   torque_workflow.bat list       - List available files
echo   torque_workflow.bat collect    - Collect torque profiles
echo   torque_workflow.bat convert    - Convert CSV to NPZ
echo   torque_workflow.bat train      - Train torque-based AMP
echo   torque_workflow.bat demo       - Run demo validation
echo.
echo Before first use, make sure to set ISAAC_PATH in this script!
echo Current ISAAC_PATH: %ISAAC_PATH%
goto end

:list_files
echo.
echo 📋 Listing available files...
%ISAAC_PATH%\python.bat example_torque_workflow.py --list
goto end

:collect_torques
echo.
echo 🔍 Available trained models:
dir /b logs\skrl\Biomech\*\checkpoints\*.pt 2>nul | findstr /i "best_agent.pt"
echo.
if "%2"=="" (
    echo Usage: torque_workflow.bat collect [model_path]
    echo Example: torque_workflow.bat collect "logs\skrl\Biomech\2025-07-10_15-12-07_Test1\checkpoints\best_agent.pt"
    goto end
)
echo 🚀 Collecting torque profiles from: %2
%ISAAC_PATH%\python.bat scripts\skrl\biomechanics.py --checkpoint "%2" --save_torque_profiles
goto end

:convert_csv
echo.
if "%2"=="" (
    echo 📊 Recent CSV files:
    dir /b /o:-d outputs\*\*\*.csv 2>nul | head -5
    echo.
    echo Usage: torque_workflow.bat convert [csv_file] [motion_name]
    echo Example: torque_workflow.bat convert "outputs\2025-07-10\16-30-45\data.csv" "my_torques"
    goto end
)
set motion_name=%3
if "%motion_name%"=="" set motion_name=collected_torques
echo 🔄 Converting CSV to NPZ: %motion_name%
%ISAAC_PATH%\python.bat Movement\csv_to_torque_motion.py --csv_file "%2" --motion_name "%motion_name%"
goto end

:train_amp
echo.
echo 🎯 Available torque motion files:
dir /b Movement\torque_motions\*.npz 2>nul
echo.
echo 🏋️ Starting torque-based AMP training...
%ISAAC_PATH%\python.bat scripts\skrl\train_torque_amp.py --use_torque_amp --num_envs 2048
goto end

:run_demo
echo.
echo 🧪 Running demo validation...
%ISAAC_PATH%\python.bat demo_torque_workflow.py --full-check
goto end

:end
echo.
echo =========================================
echo Torque workflow command completed
echo =========================================
pause
