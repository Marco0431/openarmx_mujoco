"""中文 Tkinter 启动器：手持版抓取扰动实验。

导入本模块不会创建窗口；命令行 ``--help`` 由 argparse 提供。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
RUNTIME = BASE_DIR / "exp2_nmpcc_latest_grasp_pose.py"
EXPECTED_PYTHON = Path(r"D:\acados_env\python.exe")


def _finite(name: str, value: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name}必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}必须是有限数")
    if positive and result <= 0:
        raise ValueError(f"{name}必须大于0")
    if nonnegative and result < 0:
        raise ValueError(f"{name}必须是非负数")
    return result


def make_output_dir(output_name="handheld_disturbance") -> Path:
    root = BASE_DIR / "results" / output_name
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"_{index:02d}" if index else ""
        candidate = root / f"{stamp}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("无法创建唯一输出目录")


def build_parser(scene_label="手持") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"OpenArmX {scene_label}版可选箱体/双臂扰动启动器")
    parser.add_argument("--dry-run", action="store_true", help="仅打印说明，不创建 Tk 窗口")
    return parser


class Launcher:
    def __init__(self, root: tk.Tk, *, runtime=RUNTIME, scene_label="手持",
                 output_name="handheld_disturbance", default_distance_cm="5"):
        self.root = root
        self.runtime, self.scene_label, self.output_name = runtime, scene_label, output_name
        self.proc: subprocess.Popen | None = None
        self.output_path: Path | None = None
        self.closing = False
        self.vars = {key: tk.StringVar(value=value) for key, value in {
            "target": "箱体", "link": "7", "axis": "y", "sign": "+", "force": "0.5",
            "start": "2.5", "duration": "0.2", "t_hold": "1.0", "mode": "移动", "distance_cm": default_distance_cm,
        }.items()}
        self.status = tk.StringVar(value="就绪")
        self._build()
        self._poll()

    def _build(self):
        self.root.title(f"OpenArmX {self.scene_label}版扰动实验")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=12)
        frame.grid()
        rows = [
            ("对象", "target", ("箱体", "物理左臂", "物理右臂")),
            ("连杆 (1-7)", "link", None),
            ("世界轴", "axis", ("x", "y", "z")),
            ("方向", "sign", ("+", "-")),
            ("峰值力 (N)", "force", None),
            ("施力开始时间 (s)", "start", None),
            ("单次脉冲时长 (s)", "duration", None),
            ("保持阶段时长 (s)", "t_hold", None),
            ("搬运距离 (cm)", "distance_cm", None),
        ]
        for row, (label, key, choices) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            if key == "link":
                widget = ttk.Combobox(frame, textvariable=self.vars[key], values=tuple(str(i) for i in range(1, 8)), state="readonly", width=14)
            elif choices:
                widget = ttk.Combobox(frame, textvariable=self.vars[key], values=choices, state="readonly", width=14)
            else:
                widget = ttk.Entry(frame, textvariable=self.vars[key], width=16)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            if key == "target":
                self.target_widget = widget
            if key == "link":
                self.link_widget = widget
            if key == "distance_cm":
                self.distance_widget = widget
        self.target_widget.bind("<<ComboboxSelected>>", self._target_changed)
        self._target_changed()
        ttk.Label(frame, text="场景").grid(row=9, column=0, sticky="w", padx=(0, 8), pady=3)
        self.mode_widget = ttk.Combobox(frame, textvariable=self.vars["mode"], values=("移动", "保持"), state="readonly", width=14)
        self.mode_widget.grid(row=9, column=1, sticky="ew", pady=3)
        self.mode_widget.bind("<<ComboboxSelected>>", self._mode_changed)
        self._mode_changed()
        ttk.Label(frame, text="仅推一次，到时自动撤力。世界轴施力，连杆7为末端；图中紫色标记施力时段。", foreground="#555").grid(row=10, column=0, columnspan=2, sticky="w", pady=(7, 3))
        self.viewer = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="显示 MuJoCo viewer", variable=self.viewer).grid(row=11, column=0, columnspan=2, sticky="w")
        self.python_note = ttk.Label(frame, text=self._python_message(), foreground="#8a5200")
        self.python_note.grid(row=12, column=0, columnspan=2, sticky="w")
        buttons = ttk.Frame(frame)
        buttons.grid(row=13, column=0, columnspan=2, pady=(10, 3), sticky="ew")
        self.start_button = ttk.Button(buttons, text="开始", command=self.start)
        self.start_button.pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(frame, textvariable=self.status, wraplength=420).grid(row=14, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _target_changed(self, _event=None):
        self.link_widget.configure(state="disabled" if self.vars["target"].get() == "箱体" else "readonly")

    def _mode_changed(self, _event=None):
        self.distance_widget.configure(state="disabled" if self.vars["mode"].get() == "保持" else "normal")
        if self.vars["mode"].get() == "保持" and self.vars["start"].get() == "2.5":
            self.vars["start"].set("0.8")
        elif self.vars["mode"].get() == "移动" and self.vars["start"].get() == "0.8":
            self.vars["start"].set("2.5")

    def _python_message(self):
        actual = Path(sys.executable).resolve()
        if actual != EXPECTED_PYTHON.resolve():
            return f"提示：推荐使用 {EXPECTED_PYTHON}（当前 {actual}）"
        return f"Python：{EXPECTED_PYTHON}"

    def _poll(self):
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.status.set(f"仿真已结束，返回码 {code}。输出：{self.output_path}")
            self.proc = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if self.closing:
                self.root.destroy()
                return
        self._poll_id = self.root.after(250, self._poll)

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            self.status.set("已有仿真在运行，请先停止。")
            return
        try:
            target_name = self.vars["target"].get()
            target = {"箱体": "box", "物理左臂": "left", "物理右臂": "right"}[target_name]
            link = int(self.vars["link"].get())
            if not 1 <= link <= 7:
                raise ValueError("连杆必须是1到7")
            force = _finite("峰值力", self.vars["force"].get(), positive=True)
            start = _finite("开始时间", self.vars["start"].get(), nonnegative=True)
            duration = _finite("持续时间", self.vars["duration"].get(), positive=True)
            if duration < 0.004:
                raise ValueError("持续时间至少为0.004秒（两个物理步）")
            t_hold = _finite("保持阶段时长", self.vars["t_hold"].get(), nonnegative=True)
            distance_m = 0.0
            if self.vars["mode"].get() == "保持":
                total_time = 0.5 + t_hold + 0.3
            else:
                distance_m = _finite("搬运距离", self.vars["distance_cm"].get(), positive=True) / 100.0
                # 与入口的默认 0.01 m/s 和对称加减速时序一致；厘米转换为米。
                base_time = distance_m / 0.01
                move_time = base_time + min(0.20, 0.5 * base_time) if distance_m > 1e-9 else 0.0
                total_time = max(2.4, 0.5 + t_hold + move_time + 0.4)
            if start + duration > total_time:
                raise ValueError(f"扰动结束时间不能超过本场景总时长 {total_time:.2f}s")
            if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
                raise RuntimeError(f"必须使用指定环境启动：{EXPECTED_PYTHON}")
            output = make_output_dir(self.output_name)
        except (ValueError, KeyError, RuntimeError) as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        log_path = output / "disturbance.npz"
        stop_file = output / "stop.request"
        cmd = [sys.executable, "-X", "utf8", str(self.runtime),
               "--disturb_target", target, "--disturb_link", str(link),
               "--disturb_axis", self.vars["axis"].get(), "--disturb_force",
               str(force if self.vars["sign"].get() == "+" else -force),
               "--disturb_start", str(start), "--disturb_duration", str(duration),
               "--disturb_log", str(log_path), "--disturb_stop_file", str(stop_file), "--t_hold", str(t_hold),
               "--output_dir", str(output / "simulation")]
        if target == "box":
            cmd[cmd.index("--disturb_link") + 1] = "1"
        if self.vars["mode"].get() == "保持":
            cmd.append("--hold_only")
        else:
            cmd.extend(["--y_dist", str(distance_m)])
        if not self.viewer.get():
            cmd.append("--no_viewer")
        (output / "command.json").write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8")
        stdout = (output / "stdout.log").open("w", encoding="utf-8")
        stderr = (output / "stderr.log").open("w", encoding="utf-8")
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=stdout, stderr=stderr,
                                         stdin=subprocess.DEVNULL, shell=False, creationflags=flags)
        except OSError as exc:
            stdout.close(); stderr.close()
            self.status.set(f"启动失败：{exc}")
            return
        stdout.close(); stderr.close()
        self.output_path = output
        self.status.set(f"运行中（PID {self.proc.pid}），输出：{output}")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            stop_file = self.output_path / "stop.request" if self.output_path else None
            if stop_file:
                stop_file.touch()
            self.status.set(f"已请求正常停止，正在保存 NPZ：{self.output_path}，请稍候。")

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            stop_file = self.output_path / "stop.request" if self.output_path else None
            if stop_file:
                stop_file.touch()
            self.closing = True
            self.status.set("正在正常停止并保存 NPZ，请稍候。")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            return
        self.root.after_cancel(self._poll_id)
        self.root.destroy()


def main(argv=None, *, runtime=RUNTIME, scene_label="手持", output_name="handheld_disturbance",
         default_distance_cm="5"):
    args = build_parser(scene_label).parse_args(argv)
    if args.dry_run:
        print("启动器可用；请直接运行以打开中文 Tkinter 界面。")
        return 0
    root = tk.Tk()
    Launcher(root, runtime=runtime, scene_label=scene_label, output_name=output_name,
             default_distance_cm=default_distance_cm)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
