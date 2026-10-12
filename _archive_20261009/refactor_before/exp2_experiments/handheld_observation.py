"""旧导入兼容层；实现位于 diagnostics/handheld_observation.py。"""
import sys
from diagnostics import handheld_observation as _implementation
sys.modules[__name__] = _implementation
