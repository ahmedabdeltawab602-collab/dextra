"""dextra features (Phase 4) - public facade.

Implementation split across sibling modules for maintainability:
``_features_common`` (shared helpers), ``_features_numeric`` (transform/scale),
``_features_discretize`` (bin/encode), ``_features_derive`` (dtfeats/cross/
aggfeat), ``_features_pipeline`` (featpipe). Re-exports the eight public
functions so ``from dextra.features import ...`` keeps working unchanged.
"""
from ._features_derive import aggfeat, cross, dtfeats
from ._features_discretize import bin, binize, encode
from ._features_numeric import scale, transform
from ._features_pipeline import featpipe

__all__ = [
    "transform", "scale", "bin", "binize", "encode",
    "dtfeats", "cross", "aggfeat", "featpipe",
]
