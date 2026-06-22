"""
UAV-ISAC 信道模型
论文 Section 3 — 物理层建模

核心计算:
  - LoS/NLoS 概率 (基于仰角)
  - 路径损耗 (Urban Macro 模型)
  - 通信 SINR γ_{m,k}
  - 感知 SINR SINR^s_{m,ℓ}
  - CRB (Cramér-Rao Bound)
  - 发射协方差矩阵 R_x
"""

import numpy as np
from typing import Tuple
from scipy import constants


class ISACChannel:
    """
    UAV-ISAC 物理层信道

    论文参数:
      f_c = 5.8 GHz
      B   = 20 MHz
      N_t = N_r = 8
      P_max = 30 dBm
      NF = 9 dB
    """

    def __init__(
        self,
        carrier_freq_ghz: float = 5.8,
        bandwidth_mhz: float = 20.0,
        num_antennas_tx: int = 8,
        num_antennas_rx: int = 8,
        p_max_dbm: float = 30.0,
        noise_figure_db: float = 9.0,
    ):
        self.f_c = carrier_freq_ghz * 1e9       # Hz
        self.B = bandwidth_mhz * 1e6             # Hz
        self.N_t = num_antennas_tx
        self.N_r = num_antennas_rx
        self.P_max = 10 ** ((p_max_dbm - 30) / 10)  # Watts
        self.NF = noise_figure_db

        # 噪声功率 (dBm)
        # N0 = -174 + 10*log10(B) + NF  (dBm)
        kT = -174  # dBm/Hz @ 290K
        self.noise_power_dbm = kT + 10 * np.log10(self.B) + self.NF
        self.noise_power = 10 ** ((self.noise_power_dbm - 30) / 10)  # Watts

        # 波长
        self.wavelength = constants.speed_of_light / self.f_c

        # ULA 导向矢量预计算用的 d = λ/2
        self.d_element = self.wavelength / 2

    def los_probability(self, uav_altitude: float, horizontal_dist: float) -> float:
        """
        LoS 概率模型
        基于 3GPP UMa-AV 仰角依赖模型

        Args:
            uav_altitude: UAV 高度 (m)
            horizontal_dist: 水平距离 (m)
        Returns:
            P_LoS ∈ [0, 1]
        """
        if horizontal_dist < 1e-6:
            return 1.0
        elevation_rad = np.arctan(uav_altitude / horizontal_dist)
        elevation_deg = np.degrees(elevation_rad)

        # 3GPP TR 36.777 参数 (简化)
        if elevation_deg <= 15:
            a, b = -0.5, 15.0
        else:
            a, b = -0.2, 10.0

        p_los = 1 / (1 + a * np.exp(-b * (elevation_deg - a)))
        return float(np.clip(p_los, 0.01, 0.99))

    def path_loss_db(
        self, uav_altitude: float, distance_3d: float, horizontal_dist: float
    ) -> Tuple[float, float]:
        """
        计算 LoS 和 NLoS 路径损耗 (dB)

        论文使用概率 LoS/NLoS 模型:
          PL_LoS  = 28.0 + 22*log10(d_3D) + 20*log10(f_c)   [Urban Macro LoS]
          PL_NLoS = 32.4 + 30*log10(d_3D) + 20*log10(f_c)   [Urban Macro NLoS, 带额外穿透]
          (简化, 以 1m 为参考距离)

        Returns:
            (pl_los_db, pl_nlos_db)
        """
        f_c_ghz = self.f_c / 1e9
        # Free-space path loss at 1m reference
        pl_fs_1m = 20 * np.log10(4 * np.pi / self.wavelength)

        pl_los = pl_fs_1m + 22 * np.log10(max(distance_3d, 1.0)) + 20 * np.log10(f_c_ghz)
        # NLoS 额外损耗 ~20dB
        pl_nlos = pl_fs_1m + 30 * np.log10(max(distance_3d, 1.0)) + 20 * np.log10(f_c_ghz) + 20

        return pl_los, pl_nlos

    def channel_gain(
        self, uav_pos_3d: np.ndarray, ground_pos_2d: np.ndarray
    ) -> float:
        """
        计算 UAV 到地面节点的信道增益 |h|^2

        使用概率 LoS/NLoS + 大尺度衰落

        Args:
            uav_pos_3d: [x, y, H] UAV 位置
            ground_pos_2d: [x, y] 地面节点位置
        Returns:
            信道功率增益 (线性)
        """
        diff_2d = uav_pos_3d[:2] - ground_pos_2d
        horizontal_dist = np.linalg.norm(diff_2d)
        altitude = uav_pos_3d[2]
        distance_3d = np.sqrt(horizontal_dist ** 2 + altitude ** 2)

        p_los = self.los_probability(altitude, horizontal_dist)
        pl_los_db, pl_nlos_db = self.path_loss_db(altitude, distance_3d, horizontal_dist)

        # 概率加权路径损耗
        pl_db = p_los * pl_los_db + (1 - p_los) * pl_nlos_db

        # 小尺度衰落 (Rician, K-factor 依赖仰角)
        # 简化: 单位均值瑞利 + LoS 分量
        k_factor = 10 ** ((10 - horizontal_dist / 100) / 10)  # dB → linear, 随距离衰减
        los_component = np.sqrt(k_factor / (k_factor + 1))
        nlos_component = np.sqrt(1 / (k_factor + 1))
        small_scale = np.abs(
            los_component + nlos_component * (
                np.random.randn() + 1j * np.random.randn()
            ) / np.sqrt(2)
        ) ** 2

        path_loss_linear = 10 ** (-pl_db / 10)
        return path_loss_linear * small_scale

    def compute_communication_sinr(
        self,
        uav_idx: int,
        user_idx: int,
        uav_positions: np.ndarray,      # (M, 3)
        tx_power_per_user: np.ndarray,   # (M, K) — 每用户分配功率
        association: np.ndarray,         # (M, K) — 关联矩阵
    ) -> float:
        """
        计算公式 (3-4) 中的通信 SINR γ_{m,k}

        γ_{m,k} = (p_{m,k} * |h_{m,k}|^2) /
                   (Σ_{i≠m} p_{i,k} * |h_{i,k}|^2 + N0 + I_sense)

        其中 I_sense 是感知波束干扰
        """
        uav_pos = uav_positions[uav_idx]
        # 这里 user_idx 需要从外部获取 user 位置
        # 简化: 直接传入已计算的增益
        # 完整版本由 ISACScenario 整合
        signal_power = tx_power_per_user[uav_idx, user_idx]

        # 干扰: 其他 UAV 对同一用户的泄露
        interference = 0.0
        for other_m in range(uav_positions.shape[0]):
            if other_m == uav_idx:
                continue
            interference += tx_power_per_user[other_m, user_idx]

        sinr = signal_power / (interference + self.noise_power + 1e-12)
        return float(sinr)

    def compute_sensing_sinr(
        self,
        uav_pos_3d: np.ndarray,
        target_pos_2d: np.ndarray,
        sensing_power: float,
        tx_covariance: np.ndarray,       # R_{x,m} ∈ C^{N_t×N_t}
    ) -> float:
        """
        计算感知 SINR SINR^s_{m,ℓ}

        基于回波模型 (公式 7):
          y^r_m = Σ_ℓ α_{m,ℓ} a_r(ψ) a_t^H(ψ) x_m(t-τ) + n

        感知 SINR ∝ (sensing_power * |α|^2 * N_t * N_r) /
                     (clutter + noise)
        """
        diff_2d = uav_pos_3d[:2] - target_pos_2d
        distance = np.sqrt(np.sum(diff_2d ** 2) + uav_pos_3d[2] ** 2)

        # 路径损耗 (双程 — round-trip)
        pl_db = 20 * np.log10((4 * np.pi * distance) / self.wavelength) + 20
        path_loss = 10 ** (-pl_db / 10)

        # 阵列增益
        array_gain_tx = self.N_t
        array_gain_rx = self.N_r

        signal = sensing_power * path_loss * array_gain_tx * array_gain_rx
        sinr = signal / (self.noise_power + 1e-12)
        return float(sinr)

    def compute_crb(
        self,
        uav_pos_3d: np.ndarray,
        target_pos_2d: np.ndarray,
        sensing_sinr: float,
    ) -> float:
        """
        计算 Cramér-Rao Bound (定位精度下界)

        CRB ∝ c^2 / (8π^2 * B^2 * SINR_s * N_r)
        简化模型, 论文用于约束 ε_ℓ^max
        """
        diff_2d = uav_pos_3d[:2] - target_pos_2d
        distance = np.sqrt(np.sum(diff_2d ** 2) + uav_pos_3d[2] ** 2)

        # 角度估计的 CRB (简化)
        # CRB(θ) ∝ λ² / (SINR * N_r * d²)
        crb_angle = self.wavelength ** 2 / (
            8 * np.pi ** 2 * sensing_sinr * self.N_r * self.d_element ** 2 + 1e-12
        )
        # 转换为位置 CRB
        crb_position = distance ** 2 * crb_angle
        return float(crb_position)

    def compute_tx_covariance(
        self,
        comm_beamformers: np.ndarray,     # (K, N_t) — 每用户波束成形
        sensing_beamformer: np.ndarray,   # (N_t,) — 感知波束成形
    ) -> np.ndarray:
        """
        计算发射协方差矩阵 (公式 10)

        R_x = Σ_k w_k w_k^H + w_r w_r^H ≥ 0

        这是 ISAC 耦合的核心: 通信和感知共享同一个协方差矩阵
        """
        R_x = np.zeros((self.N_t, self.N_t), dtype=np.complex128)
        for k in range(comm_beamformers.shape[0]):
            wk = comm_beamformers[k]
            R_x += np.outer(wk, wk.conj())
        R_x += np.outer(sensing_beamformer, sensing_beamformer.conj())
        return R_x

    def generate_random_beamformers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成随机初始波束成形向量 (用于 SCA-FP 重启)

        Returns:
            comm_beamformers: (K, N_t)
            sensing_beamformer: (N_t,)
        """
        # 注意: 实际中 K 是变化的, 这里只是接口
        return None, None  # 由 ISACScenario 调用时提供实际 K
