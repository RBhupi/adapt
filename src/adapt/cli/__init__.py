"""Command-line interface modules for ADAPT pipeline execution.

This package contains core execution logic, making scripts/ optional and deletable.
"""

from adapt.cli.run_nexrad import run_nexrad_pipeline

__all__ = ['run_nexrad_pipeline']
