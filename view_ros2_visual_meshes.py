#!/usr/bin/env python3
import argparse
import copy
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Windows console defaults to a legacy code page; keep generated-scene diagnostics
# printable when the caller did not explicitly choose an encoding.
if sys.platform == "win32" and not os.environ.get("PYTHONIOENCODING"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import mujoco
import mujoco.viewer
import trimesh


ROOT = Path(__file__).resolve().parent
SOURCE_URDF = ROOT / "openarmx_robot.urdf"
OUTPUT_URDF = ROOT / "openarmx_ros2_visual_mujoco.urdf"
VISUAL_MESH_DIR = ROOT / "ros2_visual_meshes"
PACKAGE_PREFIX = "package://openarmx_description/"

# ---------- 末端结构常量（单一事实来源，ik.py 从这里导入） ----------
EE_HAND_OFFSET = 0.1001   # link7 原点 -> 端面（原 hand 法兰连接面）偏移 (m)


def ensure_mujoco_compiler(root):
    mujoco_tag = root.find("mujoco")
    if mujoco_tag is None:
        mujoco_tag = ET.Element("mujoco")
        root.insert(0, mujoco_tag)

    compiler_tag = mujoco_tag.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.Element("compiler")
        mujoco_tag.append(compiler_tag)

    compiler_tag.set("strippath", "false")
    compiler_tag.set("discardvisual", "false")


def convert_package_uri(filename: str) -> str:
    if filename.startswith(PACKAGE_PREFIX):
        return str(ROOT / filename.removeprefix(PACKAGE_PREFIX))
    return filename


def parse_scale(scale_text: str) -> list[float]:
    if not scale_text:
        return [1.0, 1.0, 1.0]
    values = [float(value) for value in scale_text.split()]
    if len(values) != 3:
        return [1.0, 1.0, 1.0]
    return values


def scale_suffix(scale: list[float]) -> str:
    signs = ["n" if value < 0 else "p" for value in scale]
    return "_mirror_" + "".join(signs) if any(value < 0 for value in scale) else ""


def bake_negative_scale(mesh_or_scene, scale: list[float]):
    if not any(value < 0 for value in scale):
        return mesh_or_scene

    transform = [
        [-1.0 if scale[0] < 0 else 1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0 if scale[1] < 0 else 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 if scale[2] < 0 else 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    mesh_or_scene.apply_transform(transform)
    return mesh_or_scene


def read_dae_diffuse_colors(source_path: Path) -> list[str]:
    if source_path.suffix.lower() != ".dae":
        return []

    root = ET.parse(source_path).getroot()
    colors = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "diffuse":
            continue
        for child in element:
            if child.tag.split("}")[-1] == "color" and child.text:
                colors.append(child.text.strip())
    return colors


def convert_visual_mesh_parts(filename: str, scale: list[float]) -> list[tuple[str, str | None]]:
    source_path = Path(convert_package_uri(filename))
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    try:
        relative_source = source_path.relative_to(ROOT / "meshes")
    except ValueError:
        relative_source = Path(source_path.name)

    target_dir = VISUAL_MESH_DIR / relative_source.parent / f"{source_path.stem}{scale_suffix(scale)}"
    target_dir.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() == ".dae":
        colors = read_dae_diffuse_colors(source_path)
        mesh_or_scene = trimesh.load(source_path)
        meshes = mesh_or_scene.dump(concatenate=False) if isinstance(mesh_or_scene, trimesh.Scene) else [mesh_or_scene]
        outputs = []

        for index, mesh_part in enumerate(meshes):
            target_path = target_dir / f"{source_path.stem}{scale_suffix(scale)}_part{index}.obj"
            if not target_path.exists():
                print(f"🔄 ROS2 visual 拆分转 OBJ: {source_path.name} -> {target_path.name}")
                mesh_part = bake_negative_scale(mesh_part.copy(), scale)
                mesh_part.export(target_path)
            color = colors[index] if index < len(colors) else (colors[-1] if colors else None)
            outputs.append((str(target_path), color))

        return outputs

    target_path = target_dir / source_path.name
    if not target_path.exists():
        print(f"📁 复制 ROS2 visual mesh: {source_path.name}")
        shutil.copy2(source_path, target_path)
    return [(str(target_path), None)]


def set_visual_material(visual: ET.Element, color: str | None, name: str) -> None:
    old_material = visual.find("material")
    if old_material is not None:
        visual.remove(old_material)

    if not color:
        return

    material = ET.Element("material", {"name": name})
    ET.SubElement(material, "color", {"rgba": color})
    visual.append(material)


def strip_hand_and_fingers(root):
    """清理末端：删除两片手指的 prismatic 关节与 mesh、hand 的视觉 mesh。

    目标是无夹具直末端：link7 端面（原 hand 法兰面）直接作为顶夹面，
    由 ik.py 用 EE_HAND_OFFSET 定位。本文件只负责把 URDF 里的夹爪清掉。
    """
    for side in ("left", "right"):
        finger_joint_names = {f"openarmx_{side}_finger_joint1", f"openarmx_{side}_finger_joint2"}
        finger_link_names = {f"openarmx_{side}_left_finger", f"openarmx_{side}_right_finger"}

        for element in list(root):
            if element.tag == "joint" and element.attrib.get("name") in finger_joint_names:
                root.remove(element)
            elif element.tag == "link" and element.attrib.get("name") in finger_link_names:
                root.remove(element)

        hand_link = root.find(f"link[@name='openarmx_{side}_hand']")
        if hand_link is not None:
            for visual in list(hand_link.findall("visual")):
                hand_link.remove(visual)


def generate_visual_urdf(source_urdf: Path, output_urdf: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    ensure_mujoco_compiler(root)

    kept_collisions = 0
    visual_meshes = 0

    for link in root.findall("link"):
        # 弃用源 URDF 的简化碰撞网格（比视觉 mesh 小，视觉仍会穿箱）；
        # 实体即视觉：下面给每个视觉零件生成同名 collision（同 mesh）。
        for collision in list(link.findall("collision")):
            link.remove(collision)

        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            if filename:
                scale = parse_scale(mesh.get("scale", ""))
                mesh_parts = convert_visual_mesh_parts(filename, scale)
                link.remove(visual)
                for index, (mesh_path, color) in enumerate(mesh_parts):
                    visual_part = copy.deepcopy(visual)
                    if "name" in visual_part.attrib:
                        visual_part.set("name", f"{visual_part.attrib['name']}_part{index}")
                    part_mesh = visual_part.find("geometry/mesh")
                    part_mesh.set("filename", mesh_path)
                    part_mesh.set("scale", " ".join(str(abs(value)) for value in scale))
                    set_visual_material(visual_part, color, f"{link.attrib.get('name', 'visual')}_mat_{index}")
                    link.append(visual_part)
                    visual_meshes += 1

                    # 实体即视觉：同 mesh 再挂一个 collision，所见即所撞
                    coll = ET.SubElement(
                        link,
                        "collision",
                        {"name": f"{visual_part.attrib.get('name', 'visual')}_collision"},
                    )
                    coll_geom = ET.SubElement(ET.SubElement(coll, "geometry"), "mesh")
                    coll_geom.set("filename", mesh_path)
                    coll_geom.set("scale", " ".join(str(abs(value)) for value in scale))
                    kept_collisions += 1

    strip_hand_and_fingers(root)  # 删手指/手部视觉，圆柱末端由 ik.py 挂载

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已生成 ROS2 visual 专用 MuJoCo URDF: {output_urdf}")
    print(f"visual_meshes = {visual_meshes} | kept_collisions = {kept_collisions}")
    return output_urdf


def main():
    parser = argparse.ArgumentParser(description="View OpenArmX with ROS2 visual meshes in MuJoCo.")
    parser.add_argument("--source", type=Path, default=SOURCE_URDF, help="Source ROS2-exported URDF")
    parser.add_argument("--output", type=Path, default=OUTPUT_URDF, help="Generated MuJoCo URDF")
    parser.add_argument("--check", action="store_true", help="Only generate and compile-check, do not open viewer")
    parser.add_argument("--gravity", action="store_true", help="Enable gravity")
    args = parser.parse_args()

    output_urdf = generate_visual_urdf(args.source.resolve(), args.output.resolve())
    model = mujoco.MjModel.from_xml_path(str(output_urdf))
    data = mujoco.MjData(model)

    if not args.gravity:
        model.opt.gravity[:] = 0.0

    print("✅ MuJoCo 加载成功")
    print("nq =", model.nq, "| nv =", model.nv, "| nu =", model.nu)
    print("nbody =", model.nbody, "| ngeom =", model.ngeom, "| nmesh =", model.nmesh)

    if args.check:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
