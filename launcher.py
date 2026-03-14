import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
import socket
import time
import psutil

class WhisperSSTLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WhisperSST.is Launcher")
        self.root.geometry("400x300")
        
        # Set icon if available
        try:
            self.root.iconbitmap("whisper_icon.ico")
        except (tk.TclError, FileNotFoundError):
            pass
        
        style = ttk.Style()
        style.configure("TButton", padding=10)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="WhisperSST.is",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Icelandic Speech Recognition",
            font=("Helvetica", 10)
        )
        subtitle_label.pack(pady=5)
        
        # Status
        self.status_label = ttk.Label(
            main_frame,
            text="Ready to launch",
            font=("Helvetica", 9)
        )
        self.status_label.pack(pady=20)
        
        # Launch button
        launch_btn = ttk.Button(
            main_frame,
            text="Start WhisperSST.is",
            command=self.launch_app
        )
        launch_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate'
        )
        
        # Help button
        help_btn = ttk.Button(
            main_frame,
            text="Help & Documentation",
            command=self.open_docs
        )
        help_btn.pack(pady=5)
        
        # Version info
        version_label = ttk.Label(
            main_frame,
            text="Version 1.0.0 (Alpha)",
            font=("Helvetica", 8)
        )
        version_label.pack(pady=20)
        
        self.process = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def kill_existing_streamlit(self):
        """Kill any existing Streamlit processes."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'streamlit' in cmdline[0].lower():
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def launch_app(self):
        """Launch the Streamlit application."""
        try:
            # Check if Streamlit is already running
            if self.is_port_in_use(8501):
                self.kill_existing_streamlit()
                time.sleep(1)  # Wait for processes to clean up
            
            self.status_label.config(text="Starting application...")
            self.progress.pack(pady=10)
            self.progress.start()
            
            # Start the Streamlit server with specific arguments
            env = os.environ.copy()
            env["STREAMLIT_SERVER_PORT"] = "8501"
            env["STREAMLIT_SERVER_ADDRESS"] = "localhost"
            
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app.py",
                    "--server.port=8501",
                    "--server.address=localhost",
                    "--server.headless=true",
                    "--browser.serverAddress=localhost",
                    "--server.runOnSave=false"
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for the server to start
            self.root.after(5000, self.check_server)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start application: {str(e)}")
            self.progress.stop()
            self.progress.pack_forget()
            self.status_label.config(text="Failed to start")

    def check_server(self):
        """Check if the server has started and open the browser."""
        if self.process and self.process.poll() is None:
            # Additional check to ensure the port is actually ready
            if self.is_port_in_use(8501):
                self.status_label.config(text="Application running")
                self.progress.stop()
                self.progress.pack_forget()
                
                # Open browser
                webbrowser.open("http://localhost:8501")
                
                # Minimize launcher
                self.root.iconify()
            else:
                # Wait a bit longer and check again
                self.root.after(2000, self.check_server)
        else:
            messagebox.showerror("Error", "Failed to start server")
            self.progress.stop()
            self.progress.pack_forget()
            self.status_label.config(text="Failed to start")

    def open_docs(self):
        """Open documentation in browser."""
        webbrowser.open("https://github.com/Magnussmari/whisperSSTis#readme")

    def on_closing(self):
        """Handle window closing."""
        if self.process and self.process.poll() is None:
            if messagebox.askokcancel("Quit", "Do you want to close WhisperSST.is?"):
                self.process.terminate()
                self.kill_existing_streamlit()
                self.root.destroy()
        else:
            self.root.destroy()

    def run(self):
        """Start the launcher."""
        self.root.mainloop()

if __name__ == "__main__":
    launcher = WhisperSSTLauncher()
    launcher.run()
