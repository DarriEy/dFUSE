"""
cFUSE Command Line Interface

Entry point for the cfuse-optimize command.
"""

import sys
from pathlib import Path

def main():
    """Main entry point for cfuse-optimize command"""
    # Import optimize_basin and run its main
    # For backwards compatibility, use the script in parent directory
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    
    from optimize_basin import main as optimize_main
    optimize_main()


if __name__ == "__main__":
    main()
