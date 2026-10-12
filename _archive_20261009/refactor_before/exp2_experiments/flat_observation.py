"""旧导入兼容层；实现位于 diagnostics/flat_observation.py。"""
import sys
from diagnostics import flat_observation as _implementation
sys.modules[__name__] = _implementation
