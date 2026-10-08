"""
Structure processors for setup and sequence-proposal context.

Only trigger-controlled ligand-pose and symmetry processing is described as
rescue; ordinary docking and alignment retain their distinct roles.
Do not eagerly import optional backends here. Processors load on demand via
`StructureProcessorRegistry.get(...)` in `astrolabe/structure_processors/base.py`.
"""

from .base import BaseProcessor, StructureProcessor, StructureProcessorRegistry, register_structure_processor
