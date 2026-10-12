"""可重复外力脉冲：世界坐标施力，位置为箱体或指定连杆重心。"""
from pathlib import Path
import json
import time
import mujoco
import numpy as np


def add_arguments(parser):
    parser.add_argument('--disturb_target', choices=['box', 'left', 'right'], default=None,
                        help='启用扰动：箱体或物理左/右臂（按模型名称，不按控制器排序）')
    parser.add_argument('--disturb_link', type=int, choices=range(1, 8), default=7,
                        help='手臂受力连杆 1～7；7 为末端，箱体忽略此项')
    parser.add_argument('--disturb_axis', choices=['x', 'y', 'z'], default='y', help='世界坐标方向')
    parser.add_argument('--disturb_force', type=float, default=0.5, help='脉冲峰值 N；负数为反方向')
    parser.add_argument('--disturb_start', type=float, default=2.5, help='开始时刻，仿真秒')
    parser.add_argument('--disturb_duration', type=float, default=0.2, help='脉冲持续秒数')
    parser.add_argument('--disturb_log', default=None, help='外力 NPZ 日志路径；启用时默认自动保存')
    parser.add_argument('--disturb_stop_file', default=None, help='启动器停止请求文件；检测后正常退出并保存日志')


class ForcePulse:
    """每步临时叠加外力；积分后撤销，不累加、不清除其他物体的外力。"""
    def __init__(self, model, data, args, end_time, output_dir):
        self.model, self.data = model, data
        self.enabled = getattr(args, 'disturb_target', None) is not None
        self.base = None
        self.rows = []
        if not self.enabled:
            return
        self.start, self.duration, self.peak = args.disturb_start, args.disturb_duration, args.disturb_force
        if not np.isfinite([self.start, self.duration, self.peak]).all():
            raise ValueError('扰动参数必须为有限数')
        if self.start < 0 or self.duration < 2 * model.opt.timestep:
            raise ValueError('扰动开始时间须非负，持续时间至少为两个物理步')
        if self.start + self.duration > end_time + 1e-10:
            raise ValueError(f'扰动结束时刻超过本次仿真 {end_time:.3f}s；请提前施力或调整实验时长')
        self.body_name = ('box' if args.disturb_target == 'box'
                          else f'openarmx_{args.disturb_target}_link{args.disturb_link}')
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self.body_id <= 0:
            raise ValueError(f'找不到可施力物体：{self.body_name}')
        self.axis = 'xyz'.index(args.disturb_axis)
        self.path = (Path(args.disturb_log) if args.disturb_log else
                     Path(output_dir) / f'disturbance_{time.time_ns()}.npz')
        if self.path.suffix.lower() != '.npz':
            raise ValueError('扰动日志路径须以 .npz 结尾')
        if self.path.exists():
            raise FileExistsError(f'扰动日志已存在：{self.path}')
        self.metadata = dict(body=self.body_name, body_id=self.body_id, frame='world',
                             point='body center of mass', axis=args.disturb_axis,
                             peak_N=self.peak, start_s=self.start, duration_s=self.duration,
                             waveform='raised_cosine', timestep=float(model.opt.timestep))
        print(f'[disturbance] {self.body_name} world {args.disturb_axis}: {self.peak:g} N, '
              f'{self.start:g}～{self.start+self.duration:g}s; log={self.path}')

    def force(self, t):
        result = np.zeros(3)
        s = (t - self.start) / self.duration
        if 0 < s < 1:
            result[self.axis] = self.peak * 0.5 * (1 - np.cos(2 * np.pi * s))
        return result

    def begin_step(self, t):
        if not self.enabled:
            return
        self.release()
        self.t, self.current = float(t), self.force(t)
        self.base = self.data.xfrc_applied[self.body_id].copy()
        self.data.xfrc_applied[self.body_id, :3] += self.current

    def end_step(self):
        if self.enabled and self.base is not None:
            # Euler 的重心缓存对应本物理步起点，与日志时间及施加的力对齐。
            self.rows.append((self.t, self.current.copy(), self.data.xipos[self.body_id].copy(),
                              self.data.xfrc_applied[self.body_id].copy()))
            self.release()

    def release(self):
        if self.base is not None:
            self.data.xfrc_applied[self.body_id] = self.base
            self.base = None

    def save(self):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path,
                            t=np.asarray([r[0] for r in self.rows]),
                            force_world=np.asarray([r[1] for r in self.rows]).reshape(-1, 3),
                            point_world=np.asarray([r[2] for r in self.rows]).reshape(-1, 3),
                            total_wrench_world=np.asarray([r[3] for r in self.rows]).reshape(-1, 6),
                            metadata_json=np.array(json.dumps(self.metadata, ensure_ascii=False)))
