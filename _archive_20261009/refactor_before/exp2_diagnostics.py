"""旧导入兼容入口；实际测量代码已集中到 diagnostics/measurements.py。

直接运行只进入历史几何对账工具，日常实验请使用两版主程序。
"""
import sys
if __name__ == "__main__":
    from diagnostics.legacy_reconciliation import main
    main()
else:
    from diagnostics import measurements as _implementation
    sys.modules[__name__] = _implementation
