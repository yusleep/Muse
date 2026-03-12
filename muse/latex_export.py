"""Backward-compatible LaTeX export shim."""

from .services import latex as _latex

TEMPLATE_ROOT = _latex.TEMPLATE_ROOT
shutil = _latex.shutil
subprocess = _latex.subprocess


def export_latex_project(state, store, run_id):
    original = _latex.TEMPLATE_ROOT
    _latex.TEMPLATE_ROOT = TEMPLATE_ROOT
    try:
        return _latex.export_latex_project(state, store, run_id)
    finally:
        _latex.TEMPLATE_ROOT = original


__all__ = ["TEMPLATE_ROOT", "export_latex_project", "shutil", "subprocess"]
