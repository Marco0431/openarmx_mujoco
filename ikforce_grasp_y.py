#!/usr/bin/env python3
"""ikforce_grasp_y.py —— 沿 y 方向来回搬运版（论文实验2 构型：夹持轴 x、搬运轴 y）

与 ikforce_grasp.py 完全同源（复用其全部实现），仅把来回移动轴从 x 换成 y：
夹持力仍沿 x（垫片对顶压箱），箱子沿 y 方向（垂直夹持轴）来回搬运。

等价命令：python ikforce_grasp.py --sway_axis y ...

运行：
    python ikforce_grasp_y.py --headless --steps 8000   # 无 GUI：y 向来回搬运 + y 向 10N 扰动
    python ikforce_grasp_y.py                           # viewer 观察
    python ikforce_grasp_y.py --dist_f 30               # 30N y 扰动会超出平贴抓法抗剪极限（如实边界）
"""

import sys

from ikforce_grasp import main

if __name__ == "__main__":
    # y 向搬运 + 扰动也沿 y（10N：已验证可扛且能看出协同补偿；30N 会翻箱）
    sys.argv += ["--sway_axis", "y", "--dist_axis", "y", "--dist_f", "10"]
    main()