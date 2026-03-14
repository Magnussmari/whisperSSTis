#!/bin/bash

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Setting up dependencies for macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install PortAudio
    echo "Installing PortAudio..."
    brew install portaudio
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Setting up dependencies for Linux..."
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Using apt package manager..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
        
    elif command -v dnf &> /dev/null; then
        # Fedora
        echo "Using dnf package manager..."
        sudo dnf install -y portaudio-devel
        
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "Using pacman package manager..."
        sudo pacman -S portaudio
        
    else
        echo "Unsupported Linux distribution. Please install PortAudio manually."
        exit 1
    fi
    
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "Dependencies installed successfully!"
echo "You can now run the WhisperSST application."
