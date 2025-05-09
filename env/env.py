import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple
import subprocess
import psutil 

# 43 35

class ActionSpace(NamedTuple):
    shape: tuple
    low: np.ndarray
    high: np.ndarray
    validate: callable

class ObservationSpace(NamedTuple):
    shape: tuple

class FDTDEnv:
    """光学仿真环境"""
    def __init__(
        self,
        fdtd_path: Path = Path("C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe"),
        work_dir: Path = Path("C:/data"),
        fsp_name: str = "jones_model.fsp",
        script_name: str = "script.lsf",
        cache_dir: Path = Path("fdtd_cache"),
        max_steps: int = 500,
        violation_penalty: float = -1,
        target_wavelength_idx: int = 43 # 43/35/150
    ):
        # 路径配置
        self.fdtd_path = fdtd_path
        self.fsp_path = work_dir / fsp_name
        self.script_path = work_dir / script_name
        self.cache_dir = cache_dir
        self.obs_cache_dir = cache_dir / "obs"  # 观测结果缓存目录
        self.raw_cache_dir = cache_dir / "raw"  # 原始数据缓存目录
        

        # 文件
        self.fdtd_input_path = work_dir / "pra.txt"
        self.fdtd_output_path = work_dir / "farfile_reflection.txt"
        
        # 创建必要目录
        self.cache_dir.mkdir(exist_ok=True)
        self.obs_cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 目标波长
        self.target_wavelength_idx = target_wavelength_idx

        # 参数约束定义
        self.param_bounds = self._build_param_bounds()
        
        # 定义动作和观测空间
        self.action_space = ActionSpace(
            shape=(10,),
            low=np.zeros(10),
            high=np.ones(10),
            validate=lambda _: True  # 实际验证在_param_check
        )
        self.observation_space = ObservationSpace(shape=(10,))
        
        # 终止条件和越界惩罚
        self.max_steps = max_steps
        self.violation_penalty = violation_penalty

        # 状态初始化
        self.current_pra = None
        self.step_count = 0

    def _build_param_bounds(self) -> ActionSpace:
        """定义参数约束"""
        low = np.array([30, 40, 40, 80, 40, 40, 30, 40, 40, 30], dtype=np.float32)
        high = np.array([160, 100, 80, 340, 100, 100, 150, 340, 100, 100], dtype=np.float32)
        
        def validate(pra: np.ndarray) -> bool:
            if not np.all((low <= pra) & (pra <= high)):
                return False
            return (pra[0] + pra[3] <= 370 and 
                    pra[6] + pra[7] <= 340 and
                    pra[9] + pra[8] + pra[5] + pra[4] + pra[1] <= 370)
        
        return ActionSpace(
            shape=(10,),
            low=low,
            high=high,
            validate=validate
        )

    def _action_to_pra(self, action: np.ndarray) -> np.ndarray:
        """将归一化动作转换为仿真参数"""
        pra = action * (self.param_bounds.high - self.param_bounds.low) + self.param_bounds.low
        pra = np.round(pra / 4) * 4  # 量化到4的倍数
        return np.clip(pra, self.param_bounds.low, self.param_bounds.high)

    def _check_parameters(self, pra: np.ndarray) -> Tuple[bool, str]:
        """检查物理参数合法性"""
        if not np.all((self.param_bounds.low <= pra) & (pra <= self.param_bounds.high)):
            return False, "individual_bound_violation"
        if not self.param_bounds.validate(pra):
            return False, "combined_bound_violation"
        return True, ""

    def _generate_valid_action(self, max_attempts: int = 10) -> np.ndarray:
        """生成合法的归一化初始动作"""
        for _ in range(max_attempts):
            action = np.random.uniform(0, 1, size=10)
            pra = self._action_to_pra(action)
            if self.param_bounds.validate(pra):
                return action
        
        # 默认合法动作（对应pra=[60,44,44,84,44,44,60,44,44,44]）
        return np.array([0.2307, 0.1, 0.1, 0.015, 0.1, 0.1, 0.2307, 0.1, 0.1, 0.1])

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.step_count = 0
        init_action = self._generate_valid_action()
        state = init_action.copy()
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        # 将这次用于仿真的参数作为状态，返回给智能体，智能体根据这次的参数决定下一次尝试的参数
        state = action.copy()  

        pra = self._action_to_pra(action)
        
        # 参数检查
        is_valid, reason = self._check_parameters(pra)
        if not is_valid:
            return self._handle_failure(action, reason)
        
        # 正常仿真流程
        obs, success = self._run_simulation(pra)

        if not success:
            return self._handle_failure(action, "simulation_failed")
        
        reward = self._calculate_reward(obs)

        done = self.step_count >= self.max_steps
        info = {
            "step": self.step_count,
            "termination_reason": "max_steps_reached" if done else None,
            "pra": pra
        }
        
        return state, reward, done, info

    def _handle_failure(self, action: np.ndarray, reason: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """处理失败情况"""
        return (
            action.copy(),
            -10.0 if reason == "simulation_failed" else self.violation_penalty,  
            True,    # 立即终止
            {
                "termination_reason": reason,
                "pra": self._action_to_pra(action)
            }
        )
    
    def _run_simulation(self, pra: np.ndarray, max_retries: int = 2) -> Tuple[np.ndarray, bool]:
        """执行仿真并处理异常（自动重试机制）
        
        Args:
            pra: 参数数组
            max_retries: 最大重试次数
            
        Returns:
            tuple: (观测结果数组, 是否成功)
            
        Raises:
            RuntimeError: 超过最大重试次数仍失败
        """
        print(f'[{time.time()-_program_start_time:.2f}]', 'simulation started, pras:', pra)


        for attempt in range(max_retries + 1):  # 包括首次尝试
            # 检查缓存（每次重试前都检查）
            if (cached := self._load_cache(pra)) is not None:
                return cached
            
            # 准备输入文件
            np.savetxt(self.fdtd_input_path, pra)

            # 如果输出文件已存在，则删除
            if self.fdtd_output_path.exists():
                self.fdtd_output_path.unlink()
            
            try:
                # 启动仿真进程（记录PID以便终止）
                proc = subprocess.Popen(
                    [str(self.fdtd_path), str(self.fsp_path), "-nw", "-run", str(self.script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 等待结果（带超时）
                start_time = time.time()
                while not self.fdtd_output_path.exists():
                    if time.time() - start_time > 300:  # 5分钟超时
                        raise TimeoutError("FDTD simulation timeout")
                    if proc.poll() is not None and proc.returncode != 0:  # 进程异常退出
                        raise ValueError(f"FDTD crashed with code {proc.returncode}")
                    time.sleep(1)
                
                # 验证结果
                raw_data = np.loadtxt(self.fdtd_output_path)
                if raw_data.shape[1] < 22:
                    raise ValueError("Invalid simulation output format")
        
                print(f'[{time.time()-_program_start_time:.2f}]', 'simulation finished')

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

                print(f'[{time.time()-_program_start_time:.2f}]', 'result cached')
                
                return obs, True
            
            except (TimeoutError, ValueError) as e:
                # 终止残留进程
                self._kill_fdtd_processes()
                
                # 清理可能生成的不完整文件
                if self.fdtd_output_path.exists():
                    self.fdtd_output_path.unlink()
                
                if attempt >= max_retries:
                    return np.zeros(self.observation_space.shape), False
                
                time.sleep(1)  # 重试前短暂等待

    def _kill_fdtd_processes(self):
        """终止所有关联的FDTD进程"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'fdtd' in proc.info['name'].lower():
                    print(f'[{time.time()-_program_start_time:.2f}]', 'kill', proc)
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.5)  # 等待进程终止

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
            return self.obs_cache_dir / f"{hash_name}.npy"
        elif file_type == "raw":
            return self.raw_cache_dir / f"{hash_name}.txt"
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
        # ?
        return raw_path if raw_path.exists() else None


if __name__ == "__main__":

    _program_start_time = time.time()

    env = FDTDEnv()
    state = env.reset()
    print(f'[{time.time()-_program_start_time:.2f}]', 'init state:', state)
    
    # 随机策略
    for _ in range(10):
        print(f'[{time.time()-_program_start_time:.2f}]', 'start loop', _)
        action = env._generate_valid_action()
        state, reward, done, info = env.step(action)
        print(f'[{time.time()-_program_start_time:.2f}]', state, reward, done, info)