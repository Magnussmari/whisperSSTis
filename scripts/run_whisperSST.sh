#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check for PortAudio
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command_exists brew; then
            echo "❌ Homebrew is not installed. Please install it first:"
            echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        
        if ! brew list portaudio >/dev/null 2>&1; then
            echo "❌ PortAudio is not installed. Installing now..."
            brew install portaudio
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            if ! dpkg -l | grep -q portaudio19-dev; then
                echo "❌ PortAudio is not installed. Installing now..."
                sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio
            fi
        elif command_exists dnf; then
            if ! dnf list installed portaudio-devel >/dev/null 2>&1; then
                echo "❌ PortAudio is not installed. Installing now..."
                sudo dnf install -y portaudio-devel
            fi
        fi
    fi
    
    echo "✅ System requirements checked"
}

# Function to run the application
run_application() {
    echo "Starting WhisperSST.is..."
    
    # Make executable if needed
    chmod +x WhisperSST
    
    # Run the application
    ./WhisperSST
}

# Main execution
echo "WhisperSST.is Launcher"
echo "--------------------"

# Check requirements
check_requirements

# Run application
run_application
