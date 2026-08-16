"""Observability: structured logging, metrics, and tracing (Phase 3.5).

Modules here must stay import-cheap: `logging` is imported by both processes
at startup, and later pieces (metrics, tracing) are pulled in lazily by the
serve path so tests and the Mac dev loop never pay for exporters they do not
use.
"""
