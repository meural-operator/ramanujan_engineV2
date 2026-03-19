import os
import sys
import subprocess
import venv
import shutil

def is_windows():
    return sys.platform == "win32"

def has_nvidia_gpu():
    try:
        # Check system PATH for nvidia-smi. Extremely fast validation for NVIDIA drivers.
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return True
        return False
    except FileNotFoundError:
        return False

def setup_virtual_environment(env_dir):
    print(f"[*] Creating Python virtual environment in {env_dir}...")
    if os.path.exists(env_dir):
        print("[!] Virtual environment already exists. Using existing environment.")
    else:
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(env_dir)
    print(f"[+] Virtual environment created at {env_dir}")

def get_pip_executable(env_dir):
    if is_windows():
        return os.path.join(env_dir, "Scripts", "pip.exe")
    else:
        return os.path.join(env_dir, "bin", "pip")

def get_python_executable(env_dir):
    if is_windows():
        return os.path.join(env_dir, "Scripts", "python.exe")
    else:
        return os.path.join(env_dir, "bin", "python")

def install_dependencies(env_dir, has_gpu):
    pip_exe = get_pip_executable(env_dir)
    
    # Mathematical and core dependencies derived from the master engine
    base_packages = [
        "mpmath==1.3.0",
        "numpy==2.3.5",
        "scipy==1.17.1", 
        "sympy==1.14.0",
        "tqdm",
        "requests", # Exclusively required for Ramanujan@Home Client-Server handshakes
        "pyrebase4"
    ]
    
    print(f"[*] Installing base math dependencies: {', '.join(base_packages)}")
    subprocess.run([pip_exe, "install"] + base_packages, check=True)
    
    print("[*] Installing PyTorch subsystem...")
    if has_gpu:
        print("[+] NVIDIA GPU Detected! Installing heavy CUDA-accelerated PyTorch tensors...")
        # Leverage highest stable pip bounds for CUDA compatibility.
        subprocess.run([pip_exe, "install", "torch==2.10.0+cu130", "torchvision==0.25.0+cu130", "--index-url", "https://download.pytorch.org/whl/cu130"], check=True)
    else:
        print("[-] No NVIDIA GPU Detected. Auto-falling back to CPU-only PyTorch binaries (Lightweight)...")
        if is_windows():
            subprocess.run([pip_exe, "install", "torch==2.10.0", "torchvision==0.25.0"], check=True)
        else:
            subprocess.run([pip_exe, "install", "torch==2.10.0+cpu", "torchvision==0.25.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
            
def install_ramanujan_machine_core(env_dir):
    pip_exe = get_pip_executable(env_dir)
    print("\n[*] Fetching Core Computing Engine (V2) directly from Local Source...")
    
    # We install the exact engine via the local archive structure so the client is 100% standalone
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        subprocess.run([pip_exe, "install", repo_root], check=True)
        print("[+] Engine V2 Synced Successfully.")
    except Exception as e:
        print(f"[!] Critical Error linking engine source control: {e}")

def main():
    print("==================================================")
    print("      Ramanujan@Home - Setup & Autoinstaller      ")
    print("==================================================")
    
    # Creates an isolated compute domain within the client directory wrapper
    env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "client_env"))
    
    gpu_available = has_nvidia_gpu()
    
    setup_virtual_environment(env_dir)
    install_dependencies(env_dir, gpu_available)
    install_ramanujan_machine_core(env_dir)
    
    python_exe = get_python_executable(env_dir)
    print("==================================================")
    print(" Installation Complete! ")
    print(" You can now run the distributed node by executing: ")
    print(f" {python_exe} ramanujan_client.py")
    print("==================================================")

if __name__ == "__main__":
    main()
