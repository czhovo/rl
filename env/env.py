import os
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class FDTDEnv:
    """光学仿真环境"""
    def __init__(
        self,
        fdtd_path: Path = Path("C:/Program Files/Lumerical/v202/bin/fdtd-solutions.exe"),
        work_dir: Path = Path("D:/data"),
        fsp_name: str = "jones_model.fsp",
        script_name: str = "script.lsf",
        cache_dir: Path = Path("fdtd_cache")
    ):
        # 路径配置
        self.fdtd_path = fdtd_path
        self.fsp_path = work_dir / fsp_name
        self.script_path = work_dir / script_name
        self.cache_dir = cache_dir

        # 文件
        self.fdtd_input_path = work_dir / "pra.txt"
        self.fdtd_output_path = work_dir / "farfile_reflection.txt"
        
        # 创建必要目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # ?
        self.target_wavelength_idx = 100  # 650nm? 600+100=700?
        
        # 定义动作/观测空间
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        
        # 状态初始化
        self.current_pra = None

    def _build_action_space(self):
        """定义动作空间（连续参数）"""
        from collections import namedtuple
        Space = namedtuple('Space', ['shape', 'low', 'high'])
        return Space(
            shape=(10,),  # 假设pra有10个参数
            low=np.array([10]*10),  # 参数最小值
            high=np.array([200]*10) # 参数最大值
        )

    def _build_observation_space(self):
        """定义观测空间（仿真结果特征）"""
        from collections import namedtuple
        Space = namedtuple('Space', ['shape'])
        return Space(shape=(10,))  # eig1, eig2, r_lr, r_rl, r_rr

    def reset(self) -> np.ndarray:
        """重置环境，返回初始观测"""
        self.current_pra = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])  # 示例初始参数
        return self._run_simulation(self.current_pra)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        返回: (observation, reward, done, info)
        """
        self.current_pra = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        
        obs = self._run_simulation(self.current_pra)
        reward = self._calculate_reward(obs)
        done = False  # 持续优化任务无需自动终止
        info = {"pra": self.current_pra.copy()}
        
        return obs, reward, done, info

    def _run_simulation(self, pra: np.ndarray) -> np.ndarray:
        """执行仿真并返回观测特征"""
        # 检查缓存
        if (cached := self._load_cache(pra)) is not None:
            return cached
        
        # 执行FDTD仿真
        np.savetxt(self.fdtd_input_path, pra)
        
        """
        '"C:/Program Files/Lumerical/v202/bin/fdtd-solutions.exe" "D:/data/jones_model.fsp" -nw -run "D:/data/script.lsf"'
        """
        os.system(
            f'"{self.fdtd_path}" '
            f'"{self.fsp_path}" '
            f'-nw -run "{self.script_path}"'
        )


        # 等待并加载结果
        while not self.fdtd_output_path.exists():
            time.sleep(1)
        raw_data = np.loadtxt(self.fdtd_output_path)
        
        # 提取特征
        obs = np.array([
            raw_data[self.target_wavelength_idx, 18],  # eig1_real
            raw_data[self.target_wavelength_idx, 19],  # eig1_imag
            raw_data[self.target_wavelength_idx, 20],  # eig2_real
            raw_data[self.target_wavelength_idx, 21],  # eig2_imag
            raw_data[self.target_wavelength_idx, 10],  # r_lr_real
            raw_data[self.target_wavelength_idx, 11],  # r_lr_imag
            raw_data[self.target_wavelength_idx, 12],  # r_rl_real
            raw_data[self.target_wavelength_idx, 13],  # r_rl_imag
            raw_data[self.target_wavelength_idx, 14],  # r_rr_real
            raw_data[self.target_wavelength_idx, 15],  # r_rr_imag
        ])
        
            # 保存缓存和原始文件
        self._save_cache(pra, obs)
        self._save_raw_file(pra)
        
        return obs

    def _calculate_reward(self, obs: np.ndarray) -> float:
        """计算奖励值"""  
        assert len(obs) == 10, f"Expected 10 features, got {len(obs)}"

        eig1_real, eig1_imag, eig2_real, eig2_imag, r_lr_real, r_lr_imag, \
            r_rl_real, r_rl_imag, r_rr_real, r_rr_imag = obs
        
        eig_diff = 0.5 * (np.abs(eig1_real - eig2_real) + np.abs(eig1_imag - eig2_imag))

        r_lr_sq = r_lr_real**2 + r_lr_imag**2
        r_rl_sq = r_rl_real**2 + r_rl_imag**2
        r_rr_sq = r_rr_real**2 + r_rr_imag**2

        # 计算圆二色性（Circular Dichroism, CD）?
        denominator = max(r_lr_sq + r_rl_sq + 2*r_rr_sq, 1e-10)
        CD = np.abs(r_lr_sq - r_rl_sq) / denominator

        # 根据条件返回不同结果
        if eig_diff > 0.5 or CD < 0.2:
            return CD
        elif eig_diff > 0.02:
            return 0.5 - eig_diff + 0.6 * CD
        else:
            return 1.0 + CD
        
    def _get_cache_path(self, pra: np.ndarray, file_type: str = "obs") -> Path:
        """获取缓存路径
        Args:
            pra: 参数数组
            file_type: 文件类型，'obs'为观测结果，'raw'为原始文件
        """
        pra_str = ",".join(map(str, pra))
        hash_name = hashlib.md5(pra_str.encode()).hexdigest()
        if file_type == "obs":
            return self.cache_dir / f"{hash_name}.npy"
        elif file_type == "raw":
            return self.cache_dir / f"{hash_name}_raw.txt"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
    def _load_cache(self, pra: np.ndarray) -> Optional[np.ndarray]:
        """加载缓存观测结果"""
        path = self._get_cache_path(pra, "obs")
        return np.load(path) if path.exists() else None

    def _save_cache(self, pra: np.ndarray, obs: np.ndarray):
        """保存缓存观测结果"""
        np.save(self._get_cache_path(pra, "obs"), obs)

    def _save_raw_file(self, pra: np.ndarray):
        """保存原始仿真文件"""
        raw_path = self._get_cache_path(pra, "raw")
        if self.fdtd_output_path.exists():
            import shutil
            shutil.copy(self.fdtd_output_path, raw_path)

    def _load_raw_file(self, pra: np.ndarray) -> Optional[Path]:
        """获取缓存的原始文件路径"""
        raw_path = self._get_cache_path(pra, "raw")
        return raw_path if raw_path.exists() else None

# 使用示例（与PPO兼容）
if __name__ == "__main__":
    env = FDTDEnv()
    obs = env.reset()
    
    # 模拟随机策略
    for _ in range(10):
        action = np.random.uniform(
            low=env.action_space.low,
            high=env.action_space.high,
            size=env.action_space.shape
        )
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward:.2f} | Obs: {obs}")