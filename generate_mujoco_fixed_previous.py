#!/usr/bin/env python3
import os
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer

try:
    import trimesh
except ImportError:
    print("❌ 缺少 trimesh 库，请运行: pip install trimesh pycollada networkx scipy")
    raise SystemExit(1)

urdf_path = "./openarmx_robot.urdf"
urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

compiled_meshes_dir = os.path.join(urdf_dir, "mujoco_meshes")
os.makedirs(compiled_meshes_dir, exist_ok=True)
print(f"⏳ 正在收集、转换模型并整合至集中目录: {compiled_meshes_dir} ...")

tree = ET.parse(urdf_path)
root = tree.getroot()

mujoco_tag = root.find("mujoco")
if mujoco_tag is None:
    mujoco_tag = ET.Element("mujoco")
    root.insert(0, mujoco_tag)

compiler_tag = mujoco_tag.find("compiler")
if compiler_tag is None:
    compiler_tag = ET.Element("compiler")
    mujoco_tag.append(compiler_tag)
compiler_tag.set("meshdir", "mujoco_meshes")

ARM_MOUNT_NEW_Z = "0.66"

for joint in root.findall("joint"):
    joint_name = joint.attrib.get("name", "")
    if joint_name in [
        "openarmx_left_openarmx_body_link0_joint",
        "openarmx_right_openarmx_body_link0_joint",
    ]:
        origin = joint.find("origin")
        if origin is not None:
            xyz_str = origin.get("xyz", "0 0 0")
            xyz = xyz_str.split()
            if len(xyz) == 3:
                old_xyz = " ".join(xyz)
                xyz[2] = ARM_MOUNT_NEW_Z
                new_xyz = " ".join(xyz)
                origin.set("xyz", new_xyz)
                print(f"🔧 已下移机械臂安装位: {joint_name}")
                print(f"   原 xyz = {old_xyz}")
                print(f"   新 xyz = {new_xyz}")

for link in root.findall("link"):
    link_name = link.attrib.get("name", "")

    for element_type in ["visual", "collision"]:
        for item in link.findall(element_type):
            geom = item.find("geometry")
            if geom is None:
                continue

            mesh = geom.find("mesh")
            if mesh is not None:
                orig_filename = mesh.get("filename", "")
                if not orig_filename:
                    continue

                actual_path = orig_filename
                if actual_path.startswith("package://"):
                    parts = actual_path.split("/", 3)
                    if len(parts) >= 4:
                        actual_path = parts[3]

                if not os.path.isabs(actual_path):
                    actual_path = os.path.normpath(os.path.join(urdf_dir, actual_path))

                if not os.path.exists(actual_path):
                    print(f"⚠️ 警告: 找不到源文件 {actual_path}")
                    continue

                base_name = os.path.basename(actual_path)

                if actual_path.lower().endswith(".dae"):
                    base_name_obj = base_name[:-4] + ".obj"
                    target_path = os.path.join(compiled_meshes_dir, base_name_obj)

                    if not os.path.exists(target_path):
                        print(f"🔄 转换并提取: {base_name} -> {base_name_obj}")
                        try:
                            scene_or_mesh = trimesh.load(actual_path)
                            scene_or_mesh.export(target_path)
                        except Exception as e:
                            print(f"   ⚠️ 转换失败: {e}")

                    mesh.set("filename", base_name_obj)

                else:
                    target_path = os.path.join(compiled_meshes_dir, base_name)
                    if not os.path.exists(target_path):
                        print(f"📁 复制提取: {base_name}")
                        shutil.copy2(actual_path, target_path)

                    mesh.set("filename", base_name)

                if "body_link0" in link_name:
                    mesh.set("scale", "0.001 0.001 0.001")

                current_scale = mesh.get("scale", "")
                safe_scale = current_scale.replace("-1.0", "1.0").replace("-0.001", "0.001")
                if safe_scale:
                    mesh.set("scale", safe_scale)

        if "right_finger" in link_name:
            for item in link.findall(element_type):
                orig = item.find("origin")
                if orig is not None:
                    orig.set("rpy", "0.0 0.0 3.14159")

fixed_urdf_path = os.path.join(urdf_dir, "openarmx_mujoco_fixed.urdf")
tree.write(fixed_urdf_path, encoding="utf-8", xml_declaration=True)
print(f"✅ 已生成 MuJoCo 专用 URDF 文件: {fixed_urdf_path}")

try:
    model = mujoco.MjModel.from_xml_path(fixed_urdf_path)
    data = mujoco.MjData(model)
    print("✅ 模型加载成功！")
    print("nq =", model.nq, "| nv =", model.nv, "| nu =", model.nu)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise SystemExit(1)

print("✅ 准备打开 MuJoCo 窗口...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20

    while viewer.is_running():
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
