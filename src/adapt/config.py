"""Internal configuration defaults for ADAPT pipeline.

This is a bridge file that imports from src/expert_config.py
DO NOT EDIT - this file just enables: from adapt.config import PIPELINE_CONFIG

Author: Bhupendra Raut
"""

import sys
from pathlib import Path

# Add src directory to path to import expert_config
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from expert_config import PIPELINE_CONFIG, get_grid_kwargs, get_output_path

__all__ = ['PIPELINE_CONFIG', 'get_grid_kwargs', 'get_output_path']
