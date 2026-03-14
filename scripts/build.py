#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning build directories...")
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable...")
    
    # Basic command with essential options
    cmd = [
        'pyinstaller',
        '--onefile',  # Create a single executable
        '--name', 'WhisperSST',
        '--clean',
        '--noconfirm',
        
        # Add data files
        '--add-data', f'model_cache{os.pathsep}model_cache',
        '--add-data', f'whisperSSTis{os.pathsep}whisperSSTis',
        
        # Essential imports
        '--hidden-import', 'streamlit',
        '--hidden-import', 'sounddevice',
        '--hidden-import', 'soundfile',
        '--hidden-import', 'torch',
        '--hidden-import', 'transformers',
        
        # Entry point
        'launcher.py'
    ]
    
    # Run PyInstaller
    print("\nRunning PyInstaller with command:")
    print(" ".join(cmd))
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Check for errors
        if process.poll() != 0:
            error_output = process.stderr.read()
            print(f"\nError output:\n{error_output}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\nError during build: {str(e)}")
        return False

def create_package():
    """Create the distribution package."""
    print("\nCreating distribution package...")
    
    try:
        # Create package directory
        package_dir = Path('dist/WhisperSST_package')
        package_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy executable
        exe_name = 'WhisperSST.exe' if sys.platform.startswith('win') else 'WhisperSST'
        exe_path = Path('dist') / exe_name
        if exe_path.exists():
            shutil.copy2(exe_path, package_dir)
            print(f"✓ Copied executable: {exe_name}")
        
        # Copy documentation and run scripts
        for file in ['README.md', 'LICENSE', 'run_whisperSST.sh', 'run_whisperSST.bat']:
            if os.path.exists(file):
                shutil.copy2(file, package_dir)
                # Make shell script executable on Unix-like systems
                if file.endswith('.sh') and not sys.platform.startswith('win'):
                    os.chmod(package_dir / file, 0o755)
                print(f"✓ Copied: {file}")
        
        # Create quick start guide
        quick_start = package_dir / 'QUICK_START.txt'
        with open(quick_start, 'w', encoding='utf-8') as f:
            f.write("""WhisperSST.is - Quick Start Guide

For Windows:
1. Double-click run_whisperSST.bat
2. Wait for the launcher window to appear
3. Click "Start WhisperSST.is"
4. The application will open in your default web browser
5. Select your microphone and start transcribing!

For macOS/Linux:
1. Open Terminal in this directory
2. Run: ./run_whisperSST.sh
3. Wait for the launcher window to appear
4. Click "Start WhisperSST.is"
5. The application will open in your default web browser
6. Select your microphone and start transcribing!

For more detailed instructions, see README.md

Note: First launch may take a few minutes while loading the model.
""")
        print("✓ Created: QUICK_START.txt")
        
        return True
        
    except Exception as e:
        print(f"\nError creating package: {str(e)}")
        return False

def main():
    """Main build process."""
    try:
        print("Starting WhisperSST.is build process...")
        
        # Clean previous builds
        clean_build()
        
        # Build executable
        if not build_executable():
            print("\n❌ Build failed!")
            return False
        
        # Create distribution package
        if not create_package():
            print("\n❌ Package creation failed!")
            return False
        
        print("\n✅ Build completed successfully!")
        print("Distribution package is available in: dist/WhisperSST_package")
        return True
        
    except KeyboardInterrupt:
        print("\n\nBuild interrupted by user!")
        return False
    except Exception as e:
        print(f"\n❌ Build failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
