"""rp -- provision and drive RunPod GPU pods from the laptop.

Workflow:  rp up -> rp bootstrap -> rp run -> rp logs -> rp down
"""

__version__ = "0.1.0"


class RpError(Exception):
    """Any user-facing error. ``rp.cli.main`` prints it to stderr and exits 1."""
