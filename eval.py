
from pathlib import Path
import time
import numpy as np
from typing import Tuple, NamedTuple, List
import os
import shutil
import matplotlib.pyplot as plt

class ActionSpace(NamedTuple):
    shape: tuple
    low: np.ndarray
    high: np.ndarray
    validate: callable

class Evaluator:
    def __init__(
        self,
        fdtd_path: Path = Path("C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe"),
        work_dir: Path = Path("C:/data/eval"),
        full_fsp_name: str = "jones_model.fsp",
        full_script_name: str = "script.lsf",
        cache_dir: Path = Path("C:/data/fdtd_cache"),
        target_wavelength_idx: int = 43, # 43/35/150
        CD_threshold: float = 0.2,      
        top_n: int = 100,
    ):

        self.top_n = top_n

        # 时间
        self._start_time = time.time()

        # 路径
        self.fdtd_path = fdtd_path
        self.work_dir = work_dir
        self.full_fsp_path = work_dir / full_fsp_name
        self.full_script_path = work_dir / full_script_name
        self.cache_file = cache_dir / "reward_cache.npz"
        self.cache_file_cp = work_dir / "reward_cache.npz"
        self.fdtd_input_path = work_dir / "pra.txt"
        self.fdtd_output_path = work_dir / "farfile_reflection.txt"
        self.plot_dir = work_dir / "output_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self._copy_cache_to_workdir()

        # 参数
        self.target_wavelength_idx = target_wavelength_idx
        self.CD_threshold = CD_threshold

        self.param_bounds = self._build_param_bounds()

    def _copy_cache_to_workdir(self) -> None:
        """将缓存文件复制到工作目录"""
        shutil.copy(self.cache_file, self.work_dir)

    def _find_topn(self) -> List[Tuple[np.ndarray, float]]:
        """加载缓存文件并提取 Top-N 参数和奖励"""
        data = np.load(self.cache_file_cp)
        pras = data['keys']
        rewards = data['values']
        top_indices = np.argsort(rewards)[::-1][:self.top_n]
        return [(np.array(pras[i]), float(rewards[i])) for i in top_indices]

    def _build_param_bounds(self) -> ActionSpace:
        """定义参数约束"""
        low = np.array([30, 40, 40, 80, 40, 40, 30, 40, 40, 30], dtype=np.int32)
        high = np.array([160, 100, 80, 340, 100, 100, 150, 340, 100, 100], dtype=np.int32)
        
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
    
    def _check_parameters(self, pra: np.ndarray) -> Tuple[bool, str]:
        """检查物理参数合法性"""
        if not np.all((self.param_bounds.low <= pra) & (pra <= self.param_bounds.high)):
            return False, "individual_bound_violation"
        if not self.param_bounds.validate(pra):
            return False, "combined_bound_violation"
        return True, ""

    def _run_full_simulation(self, pra: np.ndarray) -> None:
        """执行完整仿真"""
        
        print(f'[{time.time()-self._start_time:.2f}]', 'simulation started, pras:', pra)

        np.savetxt(self.fdtd_input_path, pra)

        # 如果输出文件已存在，则删除
        if self.fdtd_output_path.exists():
            self.fdtd_output_path.unlink()
        
        # 启动仿真进程
        cmd = f'"{self.fdtd_path}" {self.full_fsp_path} -nw -run {self.full_script_path}'
        os.system(cmd)

        # 等待结果
        while not self.fdtd_output_path.exists():
            time.sleep(1)
    
        print(f'[{time.time()-self._start_time:.2f}]', 'simulation finished')


    def _get_pattern(self, pra: np.ndarray) -> np.ndarray:
        pattern = np.zeros((400, 400), dtype=int)
        
        for x in range(pattern.shape[0]):
            for y in range(pattern.shape[1]):
                if pra[0] <= x <= pra[0] + pra[2]:
                    if (pra[9] + pra[8] + pra[5] + pra[4] <= y <= 
                        pra[9] + pra[8] + pra[5] + pra[4] + pra[1]):
                        pattern[x, y] = 1
                
                if pra[0] <= x <= pra[0] + pra[3]:
                    if (pra[9] + pra[8] + pra[5] <= y <= 
                        pra[9] + pra[8] + pra[5] + pra[4]):
                        pattern[x, y] = 1
                
                if pra[6] <= x <= pra[6] + pra[7]:
                    if pra[9] <= y <= pra[9] + pra[8]:
                        pattern[x, y] = 1
        
        return pattern.T
        
    def _plot_results(self, pra: np.ndarray, reward: float) -> None:
        data = np.loadtxt(self.fdtd_output_path)
        wavelength = data[:, 1]
        
        r_lr_real = data[:, 10]
        r_lr_imag = data[:, 11]
        r_rl_real = data[:, 12]
        r_rl_imag = data[:, 13]
        r_rr_real = data[:, 14]
        r_rr_imag = data[:, 15]
        r_lr_sq = r_lr_real**2 + r_lr_imag**2
        r_rl_sq = r_rl_real**2 + r_rl_imag**2
        r_rr_sq = r_rr_real**2 + r_rr_imag**2
        eig_1_real = data[:, 18]
        eig_1_imag = data[:, 19]
        eig_2_real = data[:, 20]
        eig_2_imag = data[:, 21]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        fig.suptitle(f"Simulation Results - Parameters: {pra}, Reward: {reward:.4f}", fontsize=16)
        
        # 第一行子图
        self._plot_subfigure_1(axes[0, 0], pra)
        self._plot_subfigure_2(axes[0, 1], wavelength, eig_1_real, eig_2_real)
        self._plot_subfigure_3(axes[0, 2], wavelength, eig_1_imag, eig_2_imag)

        # 第二行子图
        self._plot_subfigure_4(axes[1, 0], wavelength, r_lr_sq, r_rl_sq, r_rr_sq)
        self._plot_subfigure_5(axes[1, 1], wavelength, r_lr_sq, r_rl_sq, r_rr_sq)
        self._plot_subfigure_6(axes[1, 2], wavelength, r_lr_sq, r_rl_sq, r_rr_sq)

        plt.tight_layout()
        save_path = self.plot_dir / f"{','.join(map(str, pra))}.png"
        plt.savefig(save_path)
        plt.close()

        print(f'[{time.time()-self._start_time:.2f}]', 'plot saved to', save_path)

    def _plot_subfigure_1(self, ax: plt.Axes, pra: np.ndarray) -> None:
        pattern = self._get_pattern(pra)
        ax.imshow(pattern, origin='lower')
        ax.axis('off')

    def _plot_subfigure_2(self, ax: plt.Axes, wavelength: np.ndarray, 
                         eig_1_real: np.ndarray, eig_2_real: np.ndarray) -> None:
        
        ax.plot(wavelength, eig_1_real, 'darkblue', label='eig_1_real')
        ax.plot(wavelength, eig_2_real, 'purple', label='eig_2_real')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(600 + self.target_wavelength_idx, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlim(600, 800)
        ax.set_ylim(min(np.min(eig_1_real), np.min(eig_2_real)), 
                    max(np.max(eig_1_real), np.max(eig_2_real)))
        
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('Real part of eigenvalue')
        
        delta = np.abs(eig_1_real[self.target_wavelength_idx] - eig_2_real[self.target_wavelength_idx])
        ax.set_title(f'Real Eigenvalue Difference: {delta:.4f}')
        ax.legend(loc='upper right')

    def _plot_subfigure_3(self, ax: plt.Axes, wavelength: np.ndarray,
                         eig_1_imag: np.ndarray, eig_2_imag: np.ndarray) -> None:
        
        ax.plot(wavelength, eig_1_imag, 'darkblue', label='eig_1_imag')
        ax.plot(wavelength, eig_2_imag, 'purple', label='eig_2_imag')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(600 + self.target_wavelength_idx, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlim(600, 800)
        ax.set_ylim(min(np.min(eig_1_imag), np.min(eig_2_imag)), 
                    max(np.max(eig_1_imag), np.max(eig_2_imag)))
        
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('Imaginary part of eigenvalue')
        
        delta = np.abs(eig_1_imag[self.target_wavelength_idx] - eig_2_imag[self.target_wavelength_idx])
        ax.set_title(f'Imag Eigenvalue Difference: {delta:.4f}')
        ax.legend(loc='upper right')

    def _plot_subfigure_4(self, ax: plt.Axes, wavelength: np.ndarray,
                         r_lr_sq: np.ndarray, r_rl_sq: np.ndarray, r_rr_sq: np.ndarray) -> None:
        
        ax.plot(wavelength, r_lr_sq, 'red', label='$|r_{LR}|^2$')
        ax.plot(wavelength, r_rl_sq, 'green', label='$|r_{RL}|^2$')
        ax.plot(wavelength, r_rr_sq, 'blue', label='$|r_{RR}|^2$')
        
        ax.axvline(600 + self.target_wavelength_idx, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlim(600, 800)
        ax.set_ylim(0, max(np.max(r_lr_sq), np.max(r_rl_sq), np.max(r_rr_sq)))
        
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('Intensity') 
        ax.legend()

    def _plot_subfigure_5(self, ax: plt.Axes, wavelength: np.ndarray,
                         r_lr_sq: np.ndarray, r_rl_sq: np.ndarray, r_rr_sq: np.ndarray) -> None:
        
        ax.plot(wavelength, 10 * np.log10(r_lr_sq), 'red', label='$|r_{LR}|^2$ (dB)')
        ax.plot(wavelength, 10 * np.log10(r_rl_sq), 'green', label='$|r_{RL}|^2$ (dB)')
        ax.plot(wavelength, 10 * np.log10(r_rr_sq), 'blue', label='$|r_{RR}|^2$ (dB)')
        
        ax.axvline(600 + self.target_wavelength_idx, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlim(600, 800)
        ymin = 10 * np.log10(np.min([r_lr_sq, r_rl_sq, r_rr_sq])) - 5
        ax.set_ylim(ymin, 0)
        
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('Spectrum (dB)')
        ax.legend()

    def _plot_subfigure_6(self, ax: plt.Axes, wavelength: np.ndarray,
                         r_lr_sq: np.ndarray, r_rl_sq: np.ndarray, r_rr_sq: np.ndarray) -> None:
        
        denominator = np.maximum(r_lr_sq + r_rl_sq + 2 * r_rr_sq, 1e-10)
        CD = np.abs(r_lr_sq - r_rl_sq) / denominator
        target_CD = CD[self.target_wavelength_idx]
        
        ax.plot(wavelength, CD, 'red', label='CD')
        
        ax.axvline(600 + self.target_wavelength_idx, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_xlim(600, 800)
        ax.set_ylim(np.min(CD), np.max(CD))
        
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('CD')
        ax.set_title(f'CD at target: {target_CD:.4f}')
        ax.legend()
        
    def evaluate(self) -> None:
        topn = self._find_topn()
        for pra, reward in topn:
            print(f'[{time.time()-self._start_time:.2f}] Evaluating pras: {pra}, reward: {reward:.4f}')
            is_valid, reason = self._check_parameters(pra)
            if not is_valid:
                print(f'invalid. Reason: {reason}')
                continue
            self._run_full_simulation(pra)
            self._plot_results(pra, reward)


if __name__ == '__main__':
    eval = Evaluator(top_n = 10)
    eval.evaluate()
