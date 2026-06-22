"""
ISAC 场景生成器
论文 Section 3 — 完整环境采样

整合 UAV 网络 + 信道模型 → 生成单个时间槽的环境样本 E(t)
为数据生成管线提供输入
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from .uav_network import UAVNetwork
from .uav_channel import ISACChannel


@dataclass
class EnvironmentSample:
    """
    单个环境样本 (对应论文中的 E^(i))

    包含:
      - 网络拓扑 (UAV/用户/目标位置)
      - 信道状态 (CSI: 所有链路的增益)
      - 通信摘要 c(t)
      - 感知摘要 r(t)
      - BEV 地图 V(t) (文本网格或图像)
    """
    sample_id: int
    q_current: np.ndarray         # UAV 位置 (M, 3)
    u_positions: np.ndarray       # 用户位置 (K, 2)
    s_positions: np.ndarray       # 目标位置 (T, 2)
    association: np.ndarray       # 当前关联 (M, K)

    # CSI: 所有 UAV-用户链路的信道增益
    channel_gains_users: np.ndarray   # (M, K) — |h_{m,k}|^2

    # 感知: 所有 UAV-目标链路的感知 SINR
    sensing_sinrs: np.ndarray         # (M, T)

    # 摘要特征
    comm_summary: Dict               # c(t)
    sensing_summary: Dict            # r(t)
    bev_grid_text: str               # V(t) — 文本格式的 BEV


class ISACScenarioGenerator:
    """
    环境样本生成器

    每个样本 = 一次随机网络初始化
    用于:
      1. SFT 数据生成 (论文 Algorithm 1)
      2. 评估测试集生成
    """

    def __init__(
        self,
        num_uavs: int = 4,
        num_users: int = 20,
        num_targets: int = 6,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        carrier_freq_ghz: float = 5.8,
        bandwidth_mhz: float = 20.0,
        num_antennas: int = 8,
        p_max_dbm: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.M = num_uavs
        self.K = num_users
        self.T = num_targets
        self.area_size = area_size

        self.channel = ISACChannel(
            carrier_freq_ghz=carrier_freq_ghz,
            bandwidth_mhz=bandwidth_mhz,
            num_antennas_tx=num_antennas,
            num_antennas_rx=num_antennas,
            p_max_dbm=p_max_dbm,
        )

        self.rng = np.random.RandomState(seed)

    def sample(self, sample_id: int) -> EnvironmentSample:
        """
        生成一个环境样本

        Returns:
            EnvironmentSample 包含完整的环境状态
        """
        # 创建临时网络
        network = UAVNetwork(
            num_uavs=self.M,
            num_users=self.K,
            num_targets=self.T,
            area_size=self.area_size,
            seed=self.rng.randint(0, 2**31 - 1),
        )

        state = network.get_state_dict()

        # ---- 计算信道增益 (M×K) ----
        channel_gains = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            uav_pos = state["uav_positions"][m]
            for k in range(self.K):
                user_pos = state["user_positions"][k]
                channel_gains[m, k] = self.channel.channel_gain(uav_pos, user_pos)

        # ---- 计算感知 SINR (M×T) ----
        sensing_sinrs = np.zeros((self.M, self.T), dtype=np.float32)
        # 简化: 每 UAV 均分功率给感知, 其余给通信
        # (实际 SCA-FP 会重新分配)
        p_sense_per_uav = self.channel.P_max * 0.3  # 30% 给感知
        for m in range(self.M):
            uav_pos = state["uav_positions"][m]
            for t in range(self.T):
                if state["target_detected"][t]:
                    target_pos = state["target_positions"][t]
                    # 简化版 tx_covariance — 使用单位矩阵缩放
                    tx_cov = np.eye(self.channel.N_t, dtype=np.complex128) * (
                        self.channel.P_max / self.channel.N_t
                    )
                    sensing_sinrs[m, t] = self.channel.compute_sensing_sinr(
                        uav_pos, target_pos, p_sense_per_uav, tx_cov
                    )

        # ---- 构造摘要 ----
        comm_summary = self._build_comm_summary(
            network, state, channel_gains
        )
        sensing_summary = self._build_sensing_summary(
            network, state, sensing_sinrs
        )
        bev_grid_text = self._build_bev_text_grid(
            network, state
        )

        return EnvironmentSample(
            sample_id=sample_id,
            q_current=state["uav_positions"].copy(),
            u_positions=state["user_positions"].copy(),
            s_positions=state["target_positions"].copy(),
            association=state["association"].copy(),
            channel_gains_users=channel_gains,
            sensing_sinrs=sensing_sinrs,
            comm_summary=comm_summary,
            sensing_summary=sensing_summary,
            bev_grid_text=bev_grid_text,
        )

    def _build_comm_summary(
        self,
        network: UAVNetwork,
        state: dict,
        channel_gains: np.ndarray,
    ) -> Dict:
        """
        构造通信摘要 c(t)

        聚合:
          - 每用户当前 SINR (基于最近关联 UAV 的 CSI)
          - 每 UAV 负载 (关联用户数)
          - 速率压力 (所需速率 vs 可达速率)
        """
        summary = {
            "per_user_sinr_db": [],
            "per_uav_load": [],
            "rate_pressure": [],
        }

        for m in range(self.M):
            load = int(state["association"][m].sum())
            summary["per_uav_load"].append(load)

        for k in range(self.K):
            # 找到最強 UAV
            best_m = int(np.argmax(channel_gains[:, k]))
            # 计算 SINR (简化: 单用户功率 = Pmax / load)
            load = max(summary["per_uav_load"][best_m], 1)
            p_per_user = self.channel.P_max / (load + 1)  # +1 for sensing beam
            sinr = channel_gains[best_m, k] * p_per_user / self.channel.noise_power
            sinr_db = 10 * np.log10(sinr + 1e-12)
            summary["per_user_sinr_db"].append(float(sinr_db))

            # 速率压力
            rate_achievable = 20e6 * np.log2(1 + sinr)  # B=20MHz
            rate_req = network.users[k].rate_requirement_bps
            summary["rate_pressure"].append(float(rate_req / max(rate_achievable, 1)))

        return summary

    def _build_sensing_summary(
        self,
        network: UAVNetwork,
        state: dict,
        sensing_sinrs: np.ndarray,
    ) -> Dict:
        """
        构造感知摘要 r(t)

        聚合:
          - 每目标的感知置信度 (最强 UAV 的 SINR)
          - 定位难度 (CRB / ε_max)
          - 未被覆盖的目标数
        """
        summary = {
            "per_target_sinr_db": [],
            "localization_difficulty": [],
            "uncovered_targets": 0,
            "best_uav_per_target": [],
        }

        for t in range(self.T):
            if not state["target_detected"][t]:
                summary["per_target_sinr_db"].append(-999.0)
                summary["best_uav_per_target"].append(-1)
                summary["localization_difficulty"].append(float("inf"))
                continue

            best_m = int(np.argmax(sensing_sinrs[:, t]))
            sinr_best = sensing_sinrs[best_m, t]
            sinr_db = 10 * np.log10(sinr_best + 1e-12)

            summary["per_target_sinr_db"].append(float(sinr_db))
            summary["best_uav_per_target"].append(best_m)

            # CRB vs ε_max
            crb = self.channel.compute_crb(
                state["uav_positions"][best_m],
                state["target_positions"][t],
                sinr_best,
            )
            crb_max = network.targets[t].crb_requirement
            summary["localization_difficulty"].append(float(crb / max(crb_max, 1e-12)))

            if sinr_db < 10:  # < Γ_s^min
                summary["uncovered_targets"] += 1

        return summary

    def _build_bev_text_grid(self, network: UAVNetwork, state: dict) -> str:
        """
        构造文本 BEV 网格 V(t)

        文本版: 将 1000×1000m 区域划分为 10×10 网格
        每个格子编码:
          - 用户密度
          - 目标密度
          - UAV 覆盖强度

        当 use_bev_image=True 时, 用 matplotlib 渲染为实际图片
        """
        grid_size = 10
        cell_w = self.area_size[0] / grid_size
        cell_h = self.area_size[1] / grid_size

        # 计数网格
        user_grid = np.zeros((grid_size, grid_size), dtype=int)
        target_grid = np.zeros((grid_size, grid_size), dtype=int)
        uav_grid = np.zeros((grid_size, grid_size), dtype=int)

        for user_pos in state["user_positions"]:
            ix = min(int(user_pos[0] / cell_w), grid_size - 1)
            iy = min(int(user_pos[1] / cell_h), grid_size - 1)
            user_grid[iy, ix] += 1

        for target_pos, detected in zip(
            state["target_positions"], state["target_detected"]
        ):
            if detected:
                ix = min(int(target_pos[0] / cell_w), grid_size - 1)
                iy = min(int(target_pos[1] / cell_h), grid_size - 1)
                target_grid[iy, ix] += 1

        for uav_pos in state["uav_positions"]:
            ix = min(int(uav_pos[0] / cell_w), grid_size - 1)
            iy = min(int(uav_pos[1] / cell_h), grid_size - 1)
            uav_grid[iy, ix] += 1

        # 格式化为文本表
        lines = ["BEV Grid (10×10, each cell = 100m×100m):"]
        lines.append("Format: [Users|Targets|UAVs]")
        lines.append("-" * 61)

        for iy in range(grid_size - 1, -1, -1):
            row_parts = []
            for ix in range(grid_size):
                cell = f"[{user_grid[iy, ix]}|{target_grid[iy, ix]}|{uav_grid[iy, ix]}]"
                row_parts.append(cell)
            lines.append(" ".join(row_parts))

        lines.append("-" * 61)
        return "\n".join(lines)
