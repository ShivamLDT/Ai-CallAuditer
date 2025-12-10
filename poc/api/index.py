"""
Vercel Serverless Function Entry Point
This file serves as the entry point for Vercel's Python runtime.
Uses Mangum to adapt FastAPI (ASGI) for serverless environments.
"""
import os
import sys
from pathlib import Path

# Add the poc directory to Python path for imports
poc_dir = Path(__file__).parent.parent
sys.path.insert(0, str(poc_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the FastAPI app
from app.main import app

# Import Mangum for ASGI adapter
from mangum import Mangum

# Create the handler for Vercel
handler = Mangum(app, lifespan="off")
