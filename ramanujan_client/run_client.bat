@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo       Ramanujan@Home - One-Click Compute Node      
echo ==================================================

:: 1. Intelligent Python 3.13 Detection & Downloader
set "PYTHON_CMD="
py -3.13 -V >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3.13"
) else (
    python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 13 else 1)" >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        echo [*] Python 3.13 not found on system.
        if not exist "python-installer.exe" (
            echo [*] Downloading Python 3.13.0 directly from python.org...
            powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe' -OutFile 'python-installer.exe'"
        )
        echo [*] Installing Python 3.13 globally for current user...
        start /wait python-installer.exe /quiet InstallAllUsers=0 Include_pip=1 PrependPath=0
        
        :: Verify installer success via py launcher
        py -3.13 -V >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON_CMD=py -3.13"
            del python-installer.exe
        ) else (
            echo [!] Python 3.13 installation failed or requires a restart.
            pause
            exit /b 1
        )
    )
)

echo [*] Enforcing Python Runtime: %PYTHON_CMD%

:: 2. Fast-check if the globally rooted virtual environment exists
if not exist "%USERPROFILE%\.ramanujan_env\Scripts\python.exe" (
    echo [*] First-time standalone setup detected. Bootstrapping AI Environment...
    %PYTHON_CMD% setup\autoinstaller.py
    if errorlevel 1 (
        echo [!] Autoinstaller failed. Please check your system Python installation.
        pause
        exit /b 1
    )
)

echo [*] Launching Client Application...
"%USERPROFILE%\.ramanujan_env\Scripts\python.exe" ramanujan_client.py

pause
