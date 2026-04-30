# OpenArmX MuJoCo V2 中文说明

本项目是一个基于 **MuJoCo** 的 OpenArmX v10 双臂机械臂本地仿真工程，目标是把 ROS2 / RViz 中的 OpenArmX 显示效果尽量完整地迁移到 MuJoCo 中，并在此基础上提供双臂控制、左右夹爪控制和左臂 IK 示例。

本工程已经将该模型导出为 URDF，并整理了所需 mesh 资源。

## 效果预览

下面这个 GIF 展示了当前 OpenArmX MuJoCo V2 场景中的双臂模型、控制面板和夹爪交互效果：

<p align="center">
  <img src="assets/openarmx_mujoco_v2_demo.gif" alt="OpenArmX MuJoCo V2 Demo" width="420" />
</p>

---

## 🎯 1. 项目目标

这个工程主要解决以下问题：

- 从 ROS2 / xacro / URDF 模型迁移到 MuJoCo 可加载模型
- 保留 OpenArmX v10 双臂机械臂的 ROS2 视觉 mesh 效果
- 自动处理 `package://openarmx_description/...` 这类 ROS 路径
- 将 `.dae` visual mesh 转换、拆分并整理到本地目录
- 处理左右臂镜像模型中的负缩放问题
- 为 MuJoCo 显式写入材质颜色，尽量还原机械臂外观
- 生成 MuJoCo runtime XML 和展示场景 XML
- 提供双臂 control 场景，支持关节滑条拖动
- 提供左右夹爪独立开合控制
- 提供左臂 IK 场景，用于拖动目标点观察逆运动学效果

---

## 🚀 2. 快速开始

<span style="color:#d9534f;"><strong>关键：</strong>建议先完成环境自检，再运行控制或 IK 场景，能显著减少排错时间。</span>


如果你使用的是 conda 环境，可以先激活自己的 MuJoCo 环境，例如：

```bash
conda activate mujoco_env
```

环境自检：

```bash
python3 -c "import mujoco, mujoco.viewer, numpy, trimesh; print('MuJoCo environment OK')"
```

查看 ROS2 visual 版本模型：

```bash
python3 view_ros2_visual_meshes.py
```

运行双臂控制场景：

```bash
python3 control.py
```

运行左臂 IK 场景：

```bash
python3 ik.py
```

只检查模型能否生成和加载，不打开窗口：

```bash
python3 control.py --check
python3 ik.py --check
```

---

## 📦 3. 推荐依赖

建议使用 Python 3.10+，并安装以下核心依赖：

- `mujoco`
- `numpy`
- `trimesh`
- `pycollada`
- `networkx`
- `scipy`
- `glfw`
- `PyOpenGL`

安装示例：

```bash
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<span style="color:#f0ad4e;"><strong>提示：</strong>如果 `.dae` 转换失败，优先检查网格转换依赖链。</span>

如果 `.dae` 转换失败，通常需要重点检查：

```bash
python3 -c "import trimesh, collada, networkx, scipy; print('mesh conversion deps OK')"
```

---

## 🗂️ 4. 项目结构

当前主要文件结构如下：

```bash
.
├── README_CN.md
├── README.md
├── control.py
├── ik.py
├── view_ros2_visual_meshes.py
├── generate_mujoco_fixed_previous.py
├── openarmx_robot.urdf
├── openarmx_v10_bimanual.urdf
├── openarmx_ros2_visual_mujoco.urdf
├── openarmx_mujoco_fixed.urdf
├── openarmx_control_runtime.xml
├── openarmx_control_scene.xml
├── openarmx_ik_runtime.xml
├── openarmx_ik_scene.xml
├── meshes/
├── mujoco_meshes/
└── ros2_visual_meshes/
```

### 🔑 关键目录

#### `meshes/`

从 ROS2 `openarmx_description` 中复制出来的原始 mesh 资源，包含：

- `meshes/body/`
- `meshes/arm/`
- `meshes/ee/`

其中 visual 资源主要是 `.dae`，collision 资源主要是 `.stl`。

#### `ros2_visual_meshes/`

由 `view_ros2_visual_meshes.py` 生成的 MuJoCo visual mesh 目录。

这个目录是当前 V2 版本最重要的视觉资源目录。脚本会将 ROS2 的 `.dae` visual mesh 拆分、转换并整理到这里，同时保留材质颜色信息。

#### `mujoco_meshes/`

早期兼容版本生成的集中式 mesh 目录，主要用于旧版 `openarmx_mujoco_fixed.urdf` 或历史脚本。当前想要最接近 ROS2 视觉效果时，优先看 `ros2_visual_meshes/`。

---

## 📘 5. 核心文件说明

### `openarmx_robot.urdf`

当前项目的主要 URDF 输入文件。

它来自 OpenArmX v10 bimanual xacro 导出结果，包含双臂、机身和左右夹爪结构。

### `openarmx_v10_bimanual.urdf`

保留的 v10 双臂导出 URDF。这个文件用于记录 ROS2 launch 对应的原始结构，便于回溯和对照。

### `openarmx_ros2_visual_mujoco.urdf`

由 `view_ros2_visual_meshes.py` 生成的 MuJoCo 专用 visual URDF。

特点：

- 优先使用 ROS2 visual mesh
- 移除 collision 几何，避免显示和加载干扰
- 将 `package://` 路径转换为本地路径
- 将 `.dae` 拆分 / 转换为 MuJoCo 更容易加载的 mesh
- 将 ROS2 中的颜色信息写入 URDF material
- 将负缩放烘焙进 mesh，避免 MuJoCo 中镜像显示异常

### `openarmx_control_runtime.xml`

`control.py` 运行时生成的 MuJoCo XML。

特点：

- 包含双臂和夹爪 actuator
- 当前 `nu=18`
  - 14 个机械臂关节 actuator
  - 4 个夹爪 finger actuator
- 5、6、7 号腕部关节已单独提高响应速度
- 机器人整体已经按当前展示需求旋转到合适朝向

### `openarmx_control_scene.xml`

`control.py` 生成的展示场景。

包含：

- 棋盘地面
- 背景墙
- 展示圆台
- studio 风格灯光
- 左右夹爪中心可视化点

### `openarmx_ik_runtime.xml`

`ik.py` 运行时生成的 MuJoCo XML，用于左臂 IK 场景。

### `openarmx_ik_scene.xml`

`ik.py` 生成的 IK 展示场景，包含目标点、夹爪中心点和左臂 IK 交互逻辑。

---

## 🛠️ 6. 核心脚本说明

### 👀 `view_ros2_visual_meshes.py`

用途：生成最接近 ROS2 / RViz 显示效果的 MuJoCo visual URDF，并打开 MuJoCo 查看。

它主要做这些事：

- 读取 `openarmx_robot.urdf`
- 解析 `package://openarmx_description/...` 路径
- 查找并复制 ROS2 visual mesh
- 读取 `.dae` 中的 diffuse 颜色
- 将 `.dae` 内部 geometry 拆分为独立 mesh part
- 转换为 MuJoCo 更容易使用的本地 mesh
- 处理左右臂镜像时的负缩放
- 移除 collision 元素，避免简化碰撞模型影响视觉
- 写出 `openarmx_ros2_visual_mujoco.urdf`
- 打开 MuJoCo viewer 预览模型

运行：

```bash
python3 view_ros2_visual_meshes.py
```

主要输出：

- `openarmx_ros2_visual_mujoco.urdf`
- `ros2_visual_meshes/`

---

### 🎮 `control.py`

用途：双臂控制和夹爪控制主场景。

它主要做这些事：

- 调用 `view_ros2_visual_meshes.py` 生成 ROS2 visual MuJoCo URDF
- 将 URDF 编译为 MuJoCo runtime XML
- 写出 `openarmx_control_runtime.xml`
- 写出 `openarmx_control_scene.xml`
- 为 14 个双臂关节加入 position actuator
- 为 4 个 finger 关节加入 position actuator
- 为 5、6、7 号腕部关节单独提高 `kp` 和 `forcerange`
- 构建更好看的展示场景
- 支持 MuJoCo 右侧 Control 面板拖动关节滑条
- 支持左右夹爪独立拖动开合

运行：

```bash
python3 control.py
```

只检查模型：

```bash
python3 control.py --check
```

当前控制说明：

- 普通机械臂关节可以在 MuJoCo 右侧 `Control` 面板拖动
- 左夹爪两个 finger 滑条内部联动
- 右夹爪两个 finger 滑条内部联动
- 拖动任意 `openarmx_left_finger...` 滑条，只影响左夹爪
- 拖动任意 `openarmx_right_finger...` 滑条，只影响右夹爪

快捷键：

- `O`：左右夹爪一起张开
- `C`：左右夹爪一起闭合
- `Z`：左夹爪张开
- `X`：左夹爪闭合
- `N`：右夹爪张开
- `M`：右夹爪闭合
- `Q`：退出

当前关节控制特点：

- 普通手臂关节：稳定优先
- 5、6、7 号腕部关节：响应速度更快
- 夹爪关节：可以通过滑条精细控制开合大小

---

### 🧠 `ik.py`

用途：左臂位置 IK 示例。

它主要做这些事：

- 生成 ROS2 visual MuJoCo runtime 模型
- 构建左臂 IK 场景
- 使用阻尼最小二乘方法求解位置 IK
- 通过红色 mocap 目标球控制末端目标位置
- 使用青色点显示夹爪中心位置
- 支持左夹爪开合
- 支持暂停、复位和调试输出

运行：

```bash
python3 ik.py
```

只检查模型：

```bash
python3 ik.py --check
```

操作：

- 鼠标拖动红色目标球：改变左臂末端目标位置
- `O`：左夹爪张开
- `C`：左夹爪闭合
- `R`：复位
- `P`：暂停 / 继续 IK
- `D`：开启 / 关闭调试输出
- `Q`：退出

---

---

## ✅ 7. 推荐使用流程

<span style="color:#0275d8;"><strong>建议：</strong>严格按 1→5 步顺序执行，可避免大多数模型与控制问题。</span>

### 1️⃣ 第一步：确认依赖

```bash
python3 -c "import mujoco, mujoco.viewer, numpy, trimesh; print('OK')"
```

### 2️⃣ 第二步：确认 ROS2 visual 模型可生成

```bash
python3 view_ros2_visual_meshes.py
```

如果模型能正常打开，说明 visual mesh、材质和路径基本可用。

### 3️⃣ 第三步：检查 control 模型

```bash
python3 control.py --check
```

正常情况下会看到类似：

```text
position actuator 数量: 18
nq=18 nv=18 nu=18 nbody=22 ngeom=52
```

### 4️⃣ 第四步：运行双臂控制场景

```bash
python3 control.py
```

重点测试：

- 右侧 Control 面板能否拖动双臂关节
- 5、6、7 号关节响应是否足够快
- 左右夹爪是否可以分别拖动
- `Z/X/N/M/O/C` 快捷键是否正常

### 5️⃣ 第五步：运行 IK 场景

```bash
python3 ik.py
```

重点测试：

- 红色目标球是否能拖动
- 左臂末端是否跟随目标球
- 左夹爪是否能开合
- 场景朝向和视觉是否符合当前展示需求

---

## ⚠️ 8. 常见问题

### ❗ 1. MuJoCo 找不到 mesh 文件

通常原因：

- 没有在项目根目录运行脚本
- `meshes/` 或 `ros2_visual_meshes/` 不完整
- URDF 中仍存在未转换的 `package://` 路径

建议先执行：

```bash
cd /home/y/Desktop/openarmx_mojuco_v2
python3 view_ros2_visual_meshes.py
```

### 🎨 2. 颜色和 ROS2 / RViz 不一致

当前 V2 版本已经尽量从 `.dae` 中读取 diffuse 颜色，并写入 MuJoCo 可使用的 material。

如果仍然不一致，常见原因包括：

- `.dae` 中材质层级复杂
- MuJoCo 与 RViz 的光照模型不同
- mesh 被拆分后部分材质没有完整映射
- 场景灯光和反射影响最终视觉观感

优先检查：

- `view_ros2_visual_meshes.py`
- `ros2_visual_meshes/`
- `openarmx_ros2_visual_mujoco.urdf`

### 🪞 3. 左右臂镜像方向异常

ROS2 URDF 中左臂 visual mesh 可能会出现类似 `scale="1.0 -1.0 1.0"` 的负缩放。

MuJoCo 对负缩放和 mesh 法线的处理可能和 RViz 不完全一致。本项目的处理方式是：

- 不直接保留负缩放
- 将镜像变换烘焙进导出的 mesh
- 在 URDF 中使用正缩放加载

相关逻辑主要在：

```text
view_ros2_visual_meshes.py
```

### 🤏 4. 夹爪不能闭合或不能拖动

请确认使用的是当前 `control.py` 生成的 runtime，而不是旧 XML。

执行：

```bash
python3 control.py --check
```

正常应看到：

```text
position actuator 数量: 18
nu=18
```

如果只有 `nu=14`，说明夹爪 actuator 没有加入，需要重新运行当前版本的 `control.py`。

### 🐢 5. 5、6、7 号关节运动慢

当前 `control.py` 已经对 5、6、7 号腕部关节单独加快：

- `WRIST_SERVO_KP = 900`
- `WRIST_ACTUATOR_FORCE = 45`

如果仍然觉得慢，可以继续调整这两个参数，但过高可能导致抖动或不稳定。

### 🧭 6. 模型在窗口里朝向不对

机器人整体朝向在 `control.py` 和 `ik.py` 中通过一个 yaw root body 处理。

当前 `control.py` 中相关参数是：

```python
ROBOT_YAW_LEFT_90 = "-1.57079632679"
```

如果需要继续调整机器人正面方向，优先改这个值，而不是只改相机视角。

---

## 📁 9. 运行产物说明

<span style="color:#5cb85c;"><strong>同步提醒：</strong>修改 URDF / mesh / 参数后，请重新运行对应脚本刷新生成物，避免旧产物干扰。</span>

常见生成文件及作用：

- `openarmx_ros2_visual_mujoco.urdf`：ROS2 visual mesh 转换后的 MuJoCo URDF
- `ros2_visual_meshes/`：拆分、转换、镜像处理后的 visual mesh 目录
- `openarmx_control_runtime.xml`：双臂控制 runtime XML
- `openarmx_control_scene.xml`：双臂控制展示场景
- `openarmx_ik_runtime.xml`：IK runtime XML
- `openarmx_ik_scene.xml`：IK 展示场景
- `MUJOCO_LOG.TXT`：MuJoCo 运行日志

如果你修改了 URDF、mesh、材质、关节参数或控制逻辑，建议重新运行对应脚本刷新生成物。

---

## 👤 作者

- **姚文昊** (Yao Wenhao)
- 公司：成都长数机器人有限公司
- 网站：https://openarmx.com/

## 🏷️ 版本

v1.0.0

## 📜 许可证

本作品采用知识共享 署名-非商业性使用-相同方式共享 4.0 国际许可协议 (CC BY-NC-SA 4.0) 进行许可。

版权所有 (c) 2026 成都长数机器人有限公司 (Chengdu Changshu Robot Co., Ltd.)

详情请参阅 [LICENSE_CN.md](LICENSE) 文件或访问：http://creativecommons.org/licenses/by-nc-sa/4.0/

---

## 📞 联系我们

### 成都长数机器人有限公司
**Chengdu Changshu Robotics Co., Ltd.**

| 联系方式 | 信息 |
|---------|------|
| 📧 邮箱 | openarmrobot@gmail.com |
| 📱 电话/微信 | +86-17746530375 |
| 🌐 官网 | <https://openarmx.com/> |
| 📍 地址 | 天津经济技术开发区西区新业八街11号华诚机械厂 |
| 👤 联系人 | 王先生 |

