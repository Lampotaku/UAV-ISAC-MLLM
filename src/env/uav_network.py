"""
UAV-ISAC 网络拓扑与状态管理
论文 Section 3 — System Model

管理:
  - M 架 UAV (3D 位置 + 天线阵列)
  - K 个地面 IoT 用户 (2D 位置)
  - T 个感知目标 (2D 位置 + 移动模型)
  - 关联矩阵 A ∈ {0,1}^{M×K}
  - 时间槽状态更新
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class UAVState:
    """单架 UAV 的完整状态"""
    idx: int
    position_3d: np.ndarray     # [x, y, H] (meters)
    velocity_2d: np.ndarray     # [vx, vy]
    serving_users: list = field(default_factory=list)

    @property
    def altitude(self) -> float:
        return self.position_3d[2]

    @property
    def position_2d(self) -> np.ndarray:
        return self.position_3d[:2]


@dataclass
class UserState:
    """地面 IoT 用户状态"""
    idx: int
    position_2d: np.ndarray     # [x, y]
    weight: float = 1.0         # ω_k (公平性权重)
    rate_requirement_bps: float = 1e6  # R_k^min
    associated_uav: int = -1    # 当前关联 UAV idx


@dataclass
class TargetState:
    """感知目标状态"""
    idx: int
    position_2d: np.ndarray     # [x, y]
    velocity_2d: np.ndarray     # [vx, vy] (用于移动)
    detected: bool = True       # 是否被检测到 (∈ T̂)
    crb_requirement: float = 0.1  # ε_ℓ^max


class UAVNetwork:
    """
    UAV-ISAC 网络环境

    论文参数默认值:
      M=4, K=20, T=6
      area: 1000×1000 m²
      H_min=50m, H_max=300m
    """

    def __init__(
        self,
        num_uavs: int = 4,
        num_users: int = 20,
        num_targets: int = 6,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        altitude_range: Tuple[float, float] = (50.0, 300.0),
        seed: Optional[int] = None,
    ):
        self.M = num_uavs
        self.K = num_users
        self.T = num_targets
        self.area_w, self.area_h = area_size
        self.H_min, self.H_max = altitude_range

        self.rng = np.random.RandomState(seed)

        # 初始化 UAV, 用户, 目标
        self.uavs: list[UAVState] = []
        self.users: list[UserState] = []
        self.targets: list[TargetState] = []

        # 当前时间槽
        self.time_slot: int = 0

        # 关联矩阵 A ∈ {0,1}^{M×K}
        self.association: np.ndarray = np.zeros((self.M, self.K), dtype=np.int32)

        self.reset()

    def reset(self) -> None:
        """随机生成一个新的网络快照"""
        self.time_slot = 0

        # --- UAV 初始化 ---
        # 均匀分布在区域内, 高度在 H_min 和 H_max 之间
        self.uavs = []
        for m in range(self.M):
            x = self.rng.uniform(0.1 * self.area_w, 0.9 * self.area_w)
            y = self.rng.uniform(0.1 * self.area_h, 0.9 * self.area_h)
            h = self.rng.uniform(self.H_min + 20, self.H_max - 20)
            vx = self.rng.uniform(-5, 5)
            vy = self.rng.uniform(-5, 5)
            self.uavs.append(UAVState(
                idx=m,
                position_3d=np.array([x, y, h], dtype=np.float32),
                velocity_2d=np.array([vx, vy], dtype=np.float32),
            ))

        # --- 用户初始化 ---
        # 聚成 2-3 个簇 (模拟 IoT 设备分布)
        self.users = []
        num_clusters = self.rng.randint(2, 4)
        cluster_centers = self.rng.uniform(
            0.1 * self.area_w, 0.9 * self.area_h,
            size=(num_clusters, 2)
        )
        for k in range(self.K):
            center = cluster_centers[self.rng.randint(num_clusters)]
            ux = center[0] + self.rng.normal(0, 50)
            uy = center[1] + self.rng.normal(0, 50)
            ux = np.clip(ux, 0, self.area_w)
            uy = np.clip(uy, 0, self.area_h)
            weight = self.rng.uniform(0.5, 2.0)
            self.users.append(UserState(
                idx=k,
                position_2d=np.array([ux, uy], dtype=np.float32),
                weight=weight,
            ))

        # --- 目标初始化 ---
        self.targets = []
        for t in range(self.T):
            tx = self.rng.uniform(0.1 * self.area_w, 0.9 * self.area_w)
            ty = self.rng.uniform(0.1 * self.area_h, 0.9 * self.area_h)
            tvx = self.rng.uniform(-3, 3)
            tvy = self.rng.uniform(-3, 3)
            self.targets.append(TargetState(
                idx=t,
                position_2d=np.array([tx, ty], dtype=np.float32),
                velocity_2d=np.array([tvx, tvy], dtype=np.float32),
                detected=self.rng.rand() > 0.2,  # 80% 被检测到
            ))

        # 初始关联: 最近 UAV 原则
        self._update_association_nearest()

    def step(self, uav_displacements: np.ndarray) -> None:
        """
        执行一个时间槽的移动

        Args:
            uav_displacements: shape (M, 3) — 每架 UAV 的 (dx, dy, dh)
               运动约束由 v_max * Δt 限制
        """
        self.time_slot += 1

        for m in range(self.M):
            self.uavs[m].position_3d += uav_displacements[m]
            # Clamp 高度
            self.uavs[m].position_3d[2] = np.clip(
                self.uavs[m].position_3d[2], self.H_min, self.H_max
            )
            # Clamp 水平位置
            self.uavs[m].position_3d[0] = np.clip(
                self.uavs[m].position_3d[0], 0, self.area_w
            )
            self.uavs[m].position_3d[1] = np.clip(
                self.uavs[m].position_3d[1], 0, self.area_h
            )

        # 移动目标
        for t in range(self.T):
            self.targets[t].position_2d += self.targets[t].velocity_2d
            # 反弹边界
            for dim in [0, 1]:
                if self.targets[t].position_2d[dim] < 0:
                    self.targets[t].position_2d[dim] *= -1
                    self.targets[t].velocity_2d[dim] *= -1
                elif self.targets[t].position_2d[dim] > (self.area_w if dim == 0 else self.area_h):
                    self.targets[t].position_2d[dim] = (
                        2 * (self.area_w if dim == 0 else self.area_h)
                        - self.targets[t].position_2d[dim]
                    )
                    self.targets[t].velocity_2d[dim] *= -1

    def _update_association_nearest(self) -> None:
        """按最近距离更新用户关联"""
        self.association = np.zeros((self.M, self.K), dtype=np.int32)
        for k, user in enumerate(self.users):
            distances = [
                np.linalg.norm(uav.position_2d - user.position_2d)
                for uav in self.uavs
            ]
            best_uav = int(np.argmin(distances))
            self.association[best_uav, k] = 1
            self.users[k].associated_uav = best_uav

    def get_state_dict(self) -> dict:
        """导出完整网络状态 (供 prompt 构造使用)"""
        return {
            "time_slot": self.time_slot,
            "uav_positions": np.array([u.position_3d for u in self.uavs]),
            "user_positions": np.array([u.position_2d for u in self.users]),
            "user_weights": np.array([u.weight for u in self.users]),
            "target_positions": np.array([t.position_2d for t in self.targets]),
            "target_detected": np.array([t.detected for t in self.targets]),
            "association": self.association.copy(),
        }

    def get_design_variables(self) -> dict:
        """
        获取当前设计变量 Ω = {Q, A, W_c, W_s}
        返回 numpy 数组
        """
        Q = np.array([u.position_3d for u in self.uavs], dtype=np.float32)
        A = self.association.astype(np.float32)
        return {"Q": Q, "A": A}
