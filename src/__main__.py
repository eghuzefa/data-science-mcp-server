"""
Entry point for running as python -m src
"""
import asyncio
from src.server import main

if __name__ == "__main__":
    asyncio.run(main())
