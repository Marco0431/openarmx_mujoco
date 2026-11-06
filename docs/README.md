# Reproduce the model previews / 重新生成模型预览

The README images are rendered from this repository's robot meshes and `ik.py`.
No generated robot artwork or manually animated joint poses are used.

README 中的图片来自仓库机器人网格与 `ik.py`，由 MuJoCo 离屏渲染。

After installing the [basic demo dependencies](../README.md#quick-start--快速开始), install Pillow in that same environment and run from the repository root:

```sh
python -m pip install Pillow
python docs/render_preview.py --work-dir ../openarmx-preview-work
```

The work directory must be outside this repository. The script copies the source
URDF, conversion helper, IK module and mesh directories into a temporary directory
there. It generates all runtime model files in that copy, removes the temporary
copy after rendering, and writes only the final images to `docs/media/`.

工作目录需在仓库之外。脚本会在其中创建临时副本，生成 URDF/XML 与转换后的网格；
渲染完成后清理副本，仅将最终图片写入 `docs/media/`，不会重写仓库中的运行时模型。

| Output / 输出 | Contents / 内容 |
| --- | --- |
| `media/openarmx-model.png` | 800 × 500 static render at the script's resting joint positions / 初始关节位姿静态图 |
| `media/openarmx-ik.gif` | 800 × 500, 100 frames at 20 fps, a 5-second loop / 100 帧、20 fps、5 秒循环 |

The animation calls `solve_position_ik` and `clamp_qpos` from the copied `ik.py`.
Both arms follow the same scripted target, with 16 IK updates per recorded frame
and an unrecorded settling cycle. It uses `mj_forward`, not `mj_step`, so playback
speed is an illustration setting, not measured controller or robot performance.
The camera and the target marker's opacity are adjusted for visibility; the robot
geometry and solver settings are unchanged. Collision avoidance is not evaluated.

动画使用原有位置 IK 与关节限位函数；两臂跟随同一自动目标，每帧计算 16 次 IK，
录制前先运行一轮目标运动。通过 `mj_forward` 更新姿态，没有运行动力学步进，
播放速度不代表控制器或机器人性能。仅为可读性调整相机和目标标记透明度，未改模型几何
或求解器参数，也未评估避碰。

Generation was checked on Windows with MuJoCo 3.12.0 and Pillow 12.3.0.
The offscreen renderer still requires working OpenGL support. For headless Linux
backend configuration, see the [official MuJoCo rendering documentation](https://mujoco.readthedocs.io/en/stable/python.html#classic-rendering).
No interactive viewer is opened by this script.

已在 Windows、MuJoCo 3.12.0、Pillow 12.3.0 环境生成并检查；离屏渲染仍需要可用的
OpenGL 环境，脚本不会打开交互式窗口。Linux 无显示环境的后端配置见上述官方文档。
