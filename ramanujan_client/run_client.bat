@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

echo ==================================================
echo       Ramanujan@Home - One-Click Compute Node      
echo ==================================================

set "PYTHON_CMD="

:: 1. Check Local AppData Python installation
if exist "%LOCALAPPDATA%\RamanujanPython\python.exe" (
    "%LOCALAPPDATA%\RamanujanPython\python.exe" -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 13 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%LOCALAPPDATA%\RamanujanPython\python.exe"
)

:: 2. Download and Install Micromamba Python 3.13 if missing
if not "%PYTHON_CMD%"=="" goto :INSTALL_DONE

echo [*] Python 3.13 isolated runtime not found.
echo [*] Downloading portable MicroMamba engine for clean installation...
if not exist "micromamba.tar.bz2" (
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile 'micromamba.tar.bz2'"
)

echo [*] Extracting MicroMamba...
tar -xf micromamba.tar.bz2
if not exist "Library\bin\micromamba.exe" (
    echo [!] Failed to extract MicroMamba container.
    pause
    exit /b 1
)

echo [*] Resolving flawless Python 3.13 environment via Conda-Forge...
Library\bin\micromamba.exe create -p "%LOCALAPPDATA%\RamanujanPython" python=3.13 pip -c conda-forge -y

if exist "%LOCALAPPDATA%\RamanujanPython\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\RamanujanPython\python.exe"
    :: Cleanup
    if exist micromamba.tar.bz2 del micromamba.tar.bz2
    if exist Library rmdir /s /q Library
    if exist info rmdir /s /q info
) else (
    echo [!] MicroMamba Python 3.13 resolution failed.
    pause
    exit /b 1
)

:INSTALL_DONE
echo [*] Enforcing Python Runtime: "%PYTHON_CMD%"

:: 4. Fast-check and Setup Virtual Environment
if not exist "%USERPROFILE%\.ramanujan_env\Scripts\python.exe" (
    echo [*] First-time standalone setup detected. Bootstrapping AI Environment...
    "%PYTHON_CMD%" setup\autoinstaller.py
    if errorlevel 1 (
        echo [!] Autoinstaller failed. Please check your system Python installation.
        pause
        exit /b 1
    )
)

echo [*] Launching Client Application...
"%USERPROFILE%\.ramanujan_env\Scripts\python.exe" ramanujan_client.py

pause
