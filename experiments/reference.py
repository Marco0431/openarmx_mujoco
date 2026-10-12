"""批量生成当前节点到终端节点的阶段参考。"""
from dataclasses import dataclass
import numpy as np
from nmpcc.data import ReferenceHorizon


def motion_timing(distance, vy):
    distance, speed = abs(float(distance)), abs(float(vy))
    if distance <= 1e-9 or speed <= 1e-9:
        return 0.0, 0.0, 0.0
    base = distance / speed
    ramp = min(0.20, 0.5 * base)
    return base, ramp, base + ramp


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s * s - 2.0 * s * s * s


@dataclass(slots=True)
class ReferenceSample:
    refs: dict
    vrefs: dict
    fmag: float
    fmag_dot: float
    phase: str
    ramp_s: float


class PhaseReference:
    """夹紧、保持、平滑搬运和停止；未来节点一次批量计算。"""
    def __init__(self, force, vy, distance, t_ramp, t_hold, press0, press, rv_ref, f_start):
        self.force, self.f_start = float(force), float(f_start)
        self.speed, self.distance = abs(float(vy)), abs(float(distance))
        self.t_ramp = max(0.0, float(t_ramp))
        self.t_hold_end = self.t_ramp + max(0.0, float(t_hold))
        self.press0 = np.array([press0[s] for s in ('left', 'right')])
        self.press = np.array([press[s] for s in ('left', 'right')])
        self.rv_ref = np.array([rv_ref[s] for s in ('left', 'right')])
        self.base_time, self.accel_time, self.move_duration = motion_timing(distance, vy)
        self.move_end = self.t_hold_end + self.move_duration

    def horizon(self, t, node_times):
        times = float(t) + np.asarray(node_times)
        z = times / self.t_ramp if self.t_ramp > 1e-9 else np.ones_like(times)
        progress = smoothstep(z)
        rate = np.zeros_like(times)
        inside = (z > 0) & (z < 1)
        if self.t_ramp > 1e-9:
            rate[inside] = 6.0 * z[inside] * (1.0 - z[inside]) / self.t_ramp
        offset, velocity = np.zeros_like(times), np.zeros_like(times)
        if self.move_duration > 0:
            tau, ramp, speed = times - self.t_hold_end, self.accel_time, self.speed
            accelerating = (tau >= 0) & (tau < ramp)
            cruising = (tau >= ramp) & (tau < self.base_time)
            decelerating = (tau >= self.base_time) & (times < self.move_end)
            za = tau[accelerating] / ramp
            offset[accelerating] = -speed * ramp * (za**3 - 0.5 * za**4)
            velocity[accelerating] = -speed * smoothstep(za)
            offset[cruising] = -speed * (tau[cruising] - 0.5 * ramp)
            velocity[cruising] = -speed
            zd = (tau[decelerating] - self.base_time) / ramp
            offset[decelerating] = -speed * (self.base_time - 0.5*ramp + ramp*(zd-zd**3+0.5*zd**4))
            velocity[decelerating] = -speed * (1.0 - smoothstep(zd))
            offset[times >= self.move_end] = -self.distance
        poses = np.zeros((len(times), 2, 6))
        velocities, forces, force_rates = np.zeros_like(poses), np.zeros_like(poses), np.zeros_like(poses)
        delta = self.press - self.press0
        poses[:, :, :3] = self.press0 + progress[:, None, None] * delta
        poses[:, :, 1] += offset[:, None]
        poses[:, :, 3:] = self.rv_ref
        velocities[:, :, :3] = rate[:, None, None] * delta
        velocities[:, :, 1] += velocity[:, None]
        forces[:, :, 0] = (self.f_start + (self.force-self.f_start)*progress)[:, None] * [-1, 1]
        force_rates[:, :, 0] = ((self.force-self.f_start)*rate)[:, None] * [-1, 1]
        return ReferenceHorizon(node_times, poses, velocities, forces, force_rates)

    def phase(self, t):
        if self.move_duration > 0 and t >= self.t_hold_end:
            return 'move'
        return 'ramp' if t < self.t_ramp else 'hold'

    def sample(self, t):
        h = self.horizon(t, np.zeros(1))
        return ReferenceSample(dict(zip(('left', 'right'), h.poses[0])),
            dict(zip(('left', 'right'), h.velocities[0])), float(h.forces[0, 1, 0]),
            float(h.force_rates[0, 1, 0]), self.phase(t),
            float(smoothstep(t / self.t_ramp)) if self.t_ramp > 1e-9 else 1.0)

    def __call__(self, t):
        r = self.sample(t)
        return r.refs, r.vrefs, r.fmag, r.phase, r.ramp_s
