"""PT-PT -> English layout-preserving PDF translator.

The package is organised as a pipeline of replaceable stages:

ingestion -> parsing -> layout/classification -> OCR -> translation
-> visual (image text) -> reconstruction -> validation

See :mod:`pt2en.pipeline.orchestrator` for how the stages are wired together.
"""

__version__ = "1.0.0"
