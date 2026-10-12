# exp2 两个实验版本

## 运行入口

| 功能 | 平放版 | 手持版 |
| --- | --- | --- |
| 实验入口 | `exp2_nmpcc_latest.py` | `exp2_nmpcc_latest_grasp_pose.py` |
| 共用控制器 | `nmpcc/controller.py` | 同一实现，场景配置不同 |
| 扰动设置界面 | `run_flat_disturbance.cmd` | `run_handheld_disturbance.cmd` |
| 使用说明 | `平放扰动使用说明.md` | `手持扰动使用说明.md` |

在本机 CMD 中可运行：

```bat
cd /d D:\RobotSim\openarmx_mujoco
D:\acados_env\python.exe exp2_nmpcc_latest.py --help
D:\acados_env\python.exe exp2_nmpcc_latest_grasp_pose.py --help
```

实际仿真去掉 `--help`，需要无界面运行时使用 `--no_viewer`。
给其他人使用时，优先发送上一级的 `nmpcc_share_20260913/`，按其中环境说明安装。

## 文件用途

- `nmpcc/`：共用类型、参数编码、运动学、预测模型、OCP 和求解器生命周期。
- `experiments/`：场景配置、参考、控制循环、记录与输出、扫描及扰动管理。
- `diagnostics/`：接触和动力学采样、关节限位记录、校准及回归检查。
- `docs/`：代码阅读、两版差异、迁移和实施记录。
- `plot_exp2_fig12.py`：两版共用的原四联图，自动绘图调用保留。
- `plot_exp2_motion_force.py`：另生成左右末端 xyz 实际速度、箱体质心总速度和各自夹持力图；读取独立的 `*motion_force.npz`，可单独重绘。
- `exp2_slender_scene.py` 与 `exp2_slender_scene.xml`：平放场景生成与模型。
- `grasp_pose_scene.xml` 与 `grasp_pose_handheld_nmpcc_runtime.xml`：手持场景与运行模型。
- `ik.py`、`ikforce.py`、`grasp_pose_static.py`、`floor_mount.py`、`view_ros2_visual_meshes.py`：场景、初始抓取和模型生成依赖；这些文件也带有演示入口，但当前实验仍调用其函数，故保留原位。
- `*.urdf`、`meshes/`、`ros2_visual_meshes/`、`openarmx_*_runtime.xml`：机器人模型、网格与运行模型。
- `results/`：仿真数据和图像；`scratch/`：既有调试资料。

控制器使用上一级 `acados_codegen/` 的短哈希缓存。共享核心使用独立模型版本，
不会加载旧两版控制器的 DLL；不同几何和关节范围仍有各自缓存。

## 归档

2026-10-09 将旧实验、探针、旧日志及旧编译输出分类移入 `_archive_20261009/`。
其中近似名称 `exp2_nmpcc_latest_grasppose_scene.py` 是另一份旧场景入口，已归档；
当前手持入口是 **`exp2_nmpcc_latest_grasp_pose.py`**。

归档旧脚本可能依赖原来的目录布局，需要恢复原路径后再运行。
完整迁移清单在上一级 `_archive_20261009/move_manifest.json`。

本次重构前的完整入口、求解器和旧接口测试保存在
`_archive_20261009/refactor_before/`。它们是数值对照资料，不参与日常运行。
新的 Python 接口使用命名数据对象；acados 所需的扁平参数只在
`nmpcc/parameters.py` 内编码，旧数组布局仅用于历史日志和回归对照。

从 `docs/代码阅读说明.md` 开始阅读；配置差异见 `docs/两版配置与测量差异.md`，
实施经验及验证结果分别在 `docs/共用核心重构记录.md`、`docs/共用核心验收结果.md`。
新增图的数据定义、采样时刻、输出位置和重绘命令见 `docs/速度与夹持力绘图.md`。
本次对照结果与既有控制限制见 `docs/速度与夹持力验收记录.md`。

已有回归检查可从 CMD 执行：

```bat
D:\acados_env\python.exe diagnostics\run_checks.py
```
