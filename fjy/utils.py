import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import platform
import time
from scipy.io import loadmat

# ==========================================
# 1. 全局绘图配置 (Styler)
# ==========================================
class PlotStyler:
    def __init__(self):
        self.configure_fonts()
        
    def configure_fonts(self):
        """根据操作系统自动配置中文字体"""
        system_name = platform.system()
        fonts = []
        if system_name == "Windows":
            fonts = ['SimHei', 'Microsoft YaHei', 'Arial']
        elif system_name == "Darwin":
            fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
        else:
            fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']
        
        plt.rcParams['font.sans-serif'] = fonts + ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 保存字体属性供 matplotlib.font_manager 使用
        from matplotlib.font_manager import FontProperties
        if len(fonts) > 0:
            self.font_prop = FontProperties(fname=None, family=fonts[0])
        else:
            self.font_prop = None

    def get_fig_ax(self, title="", figsize=(6, 4.5)):
        """统一获取 Figure 和 Axes 对象"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontsize=12, pad=10, fontproperties=self.font_prop)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax

# 实例化单例，供其他文件导入
styler = PlotStyler()

# ==========================================
# 2. 动画播放辅助函数
# ==========================================
def play_animation(draw_func, param_values, interval=0.1):
    """在 Streamlit 中播放 Matplotlib 动画"""
    placeholder = st.empty()
    for val in param_values:
        fig = draw_func(val)
        placeholder.pyplot(fig, use_container_width=False)
        time.sleep(interval)
        plt.close(fig)

# ==========================================
# 3. ISAR 数据处理与求解器
# ==========================================
@st.cache_data
def get_isar_data():
    """加载或生成 ISAR 数据"""
    try:
        data = loadmat('isar_data.mat')
        raw = data['raw']
        rd_img = data['rd_img']
        R_axis = np.squeeze(data['R_axis'])
        Fd_axis = np.squeeze(data['Fd_axis'])
    except FileNotFoundError:
        # 模拟数据生成逻辑
        N = 256
        R_axis = np.linspace(0, 100, N)
        Fd_axis = np.linspace(-50, 50, N)
        rd_img = np.zeros((N, N), dtype=complex)
        # ... (此处省略具体的模拟生成代码，保持与之前一致) ...
        targets = [(128, 128), (100, 140), (150, 110)]
        for r, c in targets: rd_img[r, c] = 100 + 0j
        raw = np.fft.ifft2(np.fft.ifftshift(rd_img), norm='ortho')
        
    return raw, rd_img, R_axis, Fd_axis

def compute_rd_image(raw_data):
    """RD 成像 (2D FFT)"""
    return np.fft.fftshift(np.fft.fft2(raw_data, norm='ortho'))

def soft_threshold(x, lambda_val):
    """软阈值算子"""
    abs_x = np.abs(x)
    scale = np.maximum(0, abs_x - lambda_val) / (abs_x + 1e-12)
    return x * scale

class CS_ISAR_Solver:
    """统一的 ISAR 稀疏求解器"""
    def __init__(self, raw_sparse, mask, max_iter=100, tol=1e-4):
        self.y = raw_sparse
        self.mask = mask
        self.Mask2D = mask[None, :]
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        
    def _A_op(self, image_x):
        return np.fft.ifft2(np.fft.ifftshift(image_x), norm='ortho') * self.Mask2D

    def _AT_op(self, data_y):
        return np.fft.fftshift(np.fft.fft2(data_y * self.Mask2D, norm='ortho'))

    def ista(self, lambda_ratio=0.05, alpha=1.0):
        x = self._AT_op(self.y) 
        lambda_val = lambda_ratio * np.max(np.abs(x))
        self.loss_history = []
        
        for i in range(self.max_iter):
            x_prev = x.copy()
            Ax = self._A_op(x)
            res = Ax - self.y
            grad = self._AT_op(res)
            z = x - alpha * grad
            x = soft_threshold(z, lambda_val * alpha)
            
            # 记录 Loss (近似)
            loss = 0.5 * np.linalg.norm(res)**2 + lambda_val * np.sum(np.abs(x))
            self.loss_history.append(loss)
            
            if np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-12) < self.tol:
                break
        return x

    def fista(self, lambda_ratio=0.05, alpha=1.0):
        x = self._AT_op(self.y)
        y_k = x.copy()
        t_k = 1.0
        lambda_val = lambda_ratio * np.max(np.abs(x))
        self.loss_history = []
        
        for i in range(self.max_iter):
            x_prev = x.copy()
            Ax = self._A_op(y_k)
            res = Ax - self.y
            grad = self._AT_op(res)
            z = y_k - alpha * grad
            x = soft_threshold(z, lambda_val * alpha)
            
            t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            y_k = x + ((t_k - 1) / t_next) * (x - x_prev)
            t_k = t_next
            
            loss = 0.5 * np.linalg.norm(res)**2 + lambda_val * np.sum(np.abs(x))
            self.loss_history.append(loss)
            
            if np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-12) < self.tol:
                break
        return x

    def admm(self, rho=1.0, lambda_ratio=0.05):
        x = self._AT_op(self.y)
        z = x.copy()
        u = np.zeros_like(x)
        lambda_val = lambda_ratio * np.max(np.abs(x))
        inv_denom_k = 1.0 / (self.Mask2D + rho)
        self.loss_history = []
        
        for i in range(self.max_iter):
            x_prev = x.copy()
            z_prev = z.copy()
            
            # x-update
            fft_zu = np.fft.ifft2(np.fft.ifftshift(z - u), norm='ortho')
            rhs_k = self.y + rho * fft_zu
            x_k = rhs_k * inv_denom_k
            x = np.fft.fftshift(np.fft.fft2(x_k, norm='ortho'))
            
            # z-update
            z = soft_threshold(x + u, lambda_val / rho)
            
            # u-update
            u = u + x - z
            
            # Loss (Primal Residual)
            self.loss_history.append(np.linalg.norm(x - z))
            
            r_norm = np.linalg.norm(x - z)
            s_norm = np.linalg.norm(rho * (z - z_prev))
            if r_norm < self.tol and s_norm < self.tol:
                break
        return z

def plot_isar_comparison(rd_img, img_fft, img_recon, algo_name, time_cost, sampling_rate):
    """统一绘图函数"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    def _plot(ax, img, title):
        mag = np.abs(img)
        db = 20 * np.log10(mag / np.max(mag) + 1e-9)
        im = ax.imshow(db, aspect='auto', cmap='jet', origin='lower', vmin=-40, vmax=0)
        ax.set_title(title, fontproperties=styler.font_prop)
        ax.set_xlabel("Doppler")
        ax.set_ylabel("Range")
        return im

    _plot(axes[0], rd_img, "1. Ground Truth (完整数据)")
    _plot(axes[1], img_fft, f"2. 直接 FFT (采样率 {sampling_rate*100:.0f}%)")
    im = _plot(axes[2], img_recon, f"3. {algo_name} 重建\n耗时: {time_cost:.3f}s")
    
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Norm Amp (dB)')
    return fig