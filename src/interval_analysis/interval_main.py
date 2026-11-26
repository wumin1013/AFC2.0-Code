"""Windowless PyInstaller entry point for the interval tool."""
from interval_analysis.interval_tool import main

if __name__ == "__main__":
    raise SystemExit(main())
