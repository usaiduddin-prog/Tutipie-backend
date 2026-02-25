"""
run.py
------
Development server entry point.

Usage:
    python run.py

For production, run uvicorn directly:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )