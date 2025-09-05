#!/usr/bin/env python3
"""
Motion AI Startup Script
Starts both the FastAPI backend and React frontend
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting Motion AI Backend (FastAPI)...")
    try:
        # Change to project root directory
        os.chdir(Path(__file__).parent)
        
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "motion_ai.api.router:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend startup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("⏹️ Backend server stopped")

def start_frontend():
    """Start the React frontend development server"""
    print("🌐 Starting Motion AI Frontend (React)...")
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # Wait a bit for backend to start
        time.sleep(3)
        
        # Start React development server
        subprocess.run([
            "npm", "start"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend startup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("⏹️ Frontend server stopped")

def main():
    """Main startup function"""
    print("🎯 Motion AI - EMG Gesture Recognition System")
    print("=" * 50)
    print("Starting both backend and frontend servers...")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:3000")
    print("=" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Motion AI system...")
        print("Thanks for using Motion AI! 🚀")

if __name__ == "__main__":
    main()