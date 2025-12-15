"""Internal configuration defaults for ``Adapt`` pipeline.

This is a bridge file that imports from src/expert_config.py
DO NOT EDIT - this file just enables: from adapt.config import PARAM_CONFIG

Author: Bhupendra Raut
"""

import sys
from pathlib import Path

# Add src directory to path to import expert_config
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from param_config import PARAM_CONFIG, get_grid_kwargs, get_output_path

__all__ = ['PARAM_CONFIG', 'get_grid_kwargs', 'get_output_path']
