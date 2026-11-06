"""Render the README's real MuJoCo model and scripted shared-target IK preview.

Requires the basic demo packages plus Pillow. Generated URDF/XML and converted
meshes are confined to a temporary copy under --work-dir, outside this repository.
This updates qpos through ik.py's position solver; it does not step dynamics.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.frames < 2 or not 1 <= args.fps <= 50:
        parser.error("Use at least two frames and an fps between 1 and 50.")

    repository = Path(__file__).resolve().parents[1]
    work_root = args.work_dir.resolve()
    if work_root == repository or repository in work_root.parents:
        parser.error("--work-dir must be outside the repository.")
    work_root.mkdir(parents=True, exist_ok=True)
    output = repository / "docs" / "media"
    output.mkdir(parents=True, exist_ok=True)
    width, height = 800, 500

    with tempfile.TemporaryDirectory(prefix="openarmx-preview-", dir=work_root) as temporary:
        staging = Path(temporary).resolve()
        # Keep temporary cleanup and all generated models inside the chosen root.
        if staging.parent != work_root:
            raise RuntimeError("Unexpected temporary directory location")
        for name in ("ik.py", "view_ros2_visual_meshes.py", "openarmx_robot.urdf"):
            shutil.copy2(repository / name, staging / name)
        for name in ("meshes", "ros2_visual_meshes"):
            shutil.copytree(repository / name, staging / name)
        sys.path.insert(0, str(staging))
        ik = importlib.import_module("ik")
        runtime = ik.generate_runtime_xml()
        scene = ik.write_scene(runtime)
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        # Keep the target legible in the indexed-color GIF; geometry is unchanged.
        model.geom("ik_target").rgba[:] = [1.0, 0.08, 0.03, 0.85]
        data = mujoco.MjData(model)
        target_id = model.body("ik_target").mocapid[0]
        arms = []
        for config in ik.ARM_CONFIGS:
            joints = [model.joint(name).id for name in config["joint_names"]]
            arm = {
                "body": model.body(config["jacobi_body"]).id,
                "joints": joints,
                "dofs": [int(model.jnt_dofadr[j]) for j in joints],
                "qpos": [int(model.jnt_qposadr[j]) for j in joints],
                "rest": config["q_rest"].copy(),
                "marker": model.body(config["viz_body"]).mocapid[0],
            }
            data.qpos[arm["qpos"]] = arm["rest"]
            ik.clamp_qpos(model, data, arm["qpos"], joints)
            arms.append(arm)
        mujoco.mj_forward(model, data)
        center = ik.clamp_target(np.mean([
            ik.get_ee_tip(data, arm["body"]).copy() for arm in arms
        ], axis=0))

        def set_target(phase: float) -> None:
            offset = np.array([0.06 * np.cos(phase), 0.14 * np.sin(phase), 0.08 * np.sin(phase)])
            data.mocap_pos[target_id] = ik.clamp_target(center + offset)

        def solve_once() -> None:
            data.qvel[:] = 0
            data.qacc[:] = 0
            mujoco.mj_forward(model, data)
            solutions = []
            for arm in arms:
                dq, _ = ik.solve_position_ik(
                    model, data, arm["body"], data.mocap_pos[target_id],
                    ik.get_ee_tip(data, arm["body"]), arm["dofs"], arm["qpos"], arm["rest"],
                )
                solutions.append((arm, dq))
            for arm, dq in solutions:
                data.qpos[arm["qpos"]] += dq
                ik.clamp_qpos(model, data, arm["qpos"], arm["joints"])
            mujoco.mj_forward(model, data)
            for arm in arms:
                data.mocap_pos[arm["marker"]] = ik.get_ee_tip(data, arm["body"])
            mujoco.mj_forward(model, data)

        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, -0.10, 0.42]
        camera.distance = 1.45
        camera.elevation = -15
        camera.azimuth = 90

        frames = []
        qpos = []
        set_target(0)
        # Renderer creates a hidden offscreen OpenGL context, not a viewer.
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for arm in arms:
                data.mocap_pos[arm["marker"]] = ik.get_ee_tip(data, arm["body"])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render().copy()).save(output / "openarmx-model.png")
            for _ in range(240):
                solve_once()
            # One unrecorded cycle reduces the initial settling transient.
            for frame_number in range(args.frames):
                set_target(2 * np.pi * frame_number / args.frames)
                for _ in range(16):
                    solve_once()
            for frame_number in range(args.frames):
                set_target(2 * np.pi * frame_number / args.frames)
                for _ in range(16):
                    solve_once()
                renderer.update_scene(data, camera=camera)
                frames.append(Image.fromarray(renderer.render().copy()))
                qpos.append(data.qpos.copy())

        # A shared palette reduces color flicker and keeps the GIF compact.
        samples = frames[::max(1, args.frames // 12)]
        palette_sheet = Image.new("RGB", (200 * len(samples), 125))
        for index, frame in enumerate(samples):
            palette_sheet.paste(frame.resize((200, 125)), (200 * index, 0))
        palette = palette_sheet.quantize(colors=256)
        encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        encoded[0].save(
            output / "openarmx-ik.gif", save_all=True, append_images=encoded[1:],
            duration=round(1000 / args.fps), loop=0, optimize=True,
        )
        print(json.dumps({
            "mujoco": mujoco.__version__, "dimensions": [width, height],
            "frames": args.frames, "requested_fps": args.fps,
            "nq": model.nq, "nv": model.nv, "center": center.tolist(),
            "max_joint_excursion_rad": float(np.ptp(np.stack(qpos), axis=0).max()),
            "outputs": {name: (output / name).stat().st_size for name in
                        ["openarmx-model.png", "openarmx-ik.gif"]},
        }, indent=2))


if __name__ == "__main__":
    main()
