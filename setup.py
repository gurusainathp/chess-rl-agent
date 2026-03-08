"""
setup.py
--------
Makes chess-rl-agent an installable Python package.

Install in editable (development) mode so any code changes are
reflected immediately without reinstalling:

    pip install -e .

After installation, import from anywhere like:

    from chess_rl_agent.environment.chess_env import ChessEnv
    from chess_rl_agent.models.policy_network import PolicyNetwork
"""

from setuptools import setup, find_packages

setup(
    # ------------------------------------------------------------------
    # Package identity
    # ------------------------------------------------------------------
    name="chess-rl-agent",
    version="0.1.0",
    description="A reinforcement learning chess agent that learns from scratch via self-play.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/chess-rl-agent",
    license="MIT",

    # ------------------------------------------------------------------
    # Package discovery
    # ------------------------------------------------------------------
    # Tells setuptools to look inside src/ for packages.
    # Every sub-folder with an __init__.py is picked up automatically.
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # ------------------------------------------------------------------
    # Runtime dependencies
    # ------------------------------------------------------------------
    install_requires=[
        "python-chess",   # Chess board, move validation, PGN support
        "torch",          # Neural network training
        "numpy",          # Array operations and board encoding
        "pandas",         # Training log storage and analysis
        "matplotlib",     # Training curves and result plots
        "tqdm",           # Progress bars for training loops
        "streamlit",      # Interactive web UI
        "stockfish",      # Python wrapper for Stockfish engine
    ],

    # ------------------------------------------------------------------
    # Optional / development dependencies
    # ------------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest",          # Test runner
            "pytest-cov",      # Coverage reports
        ],
        "tracking": [
            "tensorboard",     # Optional experiment tracking dashboard
        ],
    },

    # ------------------------------------------------------------------
    # Python version requirement
    # ------------------------------------------------------------------
    python_requires=">=3.10",

    # ------------------------------------------------------------------
    # CLI entry points (optional — lets you run scripts from the terminal)
    # ------------------------------------------------------------------
    # After pip install -e . you can type these commands directly:
    #   chess-train
    #   chess-evaluate
    #   chess-play
    entry_points={
        "console_scripts": [
            "chess-train=chess_rl_agent.scripts.train:main",
            "chess-evaluate=chess_rl_agent.scripts.evaluate:main",
            "chess-play=chess_rl_agent.scripts.play:main",
        ],
    },

    # ------------------------------------------------------------------
    # Metadata classifiers (used by PyPI)
    # ------------------------------------------------------------------
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)