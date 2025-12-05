import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import styler, compute_rd_image, plot_isar_comparison, get_isar_data

# ==========================================
# 0. 专用求解器定义 (Huber Solver)
# ==========================================
class HuberISARSolver:
    """
    包含 Huber Smoothing 算法的 ISAR 求解器
    """
    def __init__(self, raw_sparse, mask, max_iter=100, tol=1e-5):
        self.y = raw_sparse
        self.mask = mask
        self.Mask2D = mask[None, :]
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        
    def _A_op(self, image_x):
        """正向算子: Image -> Data (Undersampled)"""
        # Image (RD Domain) -> Data (Time/Freq Domain)
        return np.fft.ifft2(np.fft.ifftshift(image_x), norm='ortho') * self.Mask2D

    def _AT_op(self, data_y):
        """伴随算子: Data -> Image"""
        # Data -> Image
        return np.fft.fftshift(np.fft.fft2(data_y * self.Mask2D, norm='ortho'))

    def _grad_huber(self, x, mu):
        """计算 Huber 平滑项的梯度"""
        abs_x = np.abs(x)
        grad = np.zeros_like(x)
        
        # Case 1: |x| <= mu (二次函数区域)
        mask_small = abs_x <= mu
        grad[mask_small] = x[mask_small] / mu
        
        # Case 2: |x| > mu (线性区域，类似 L1)
        mask_large = ~mask_small
        grad[mask_large] = x[mask_large] / (abs_x[mask_large] + 1e-15)
        
        return grad

    def huber_gradient_descent(self, lambda_ratio=0.05, alpha=1.0, mu=1e-2):
        """基础 Huber 梯度下降"""
        x = self._AT_op(self.y)
        lambda_val = lambda_ratio * np.max(np.abs(x))
        self.loss_history = []
        
        for i in range(self.max_iter):
            x_prev = x.copy()
            
            # 1. 数据拟合项梯度
            Ax = self._A_op(x)
            res = Ax - self.y
            grad_data = self._AT_op(res)
            
            # 2. 正则项梯度 (Huber)
            grad_reg = lambda_val * self._grad_huber(x, mu)
            
            # 3. 梯度下降更新
            x = x - alpha * (grad_data + grad_reg)
            
            # 记录
            loss = 0.5 * np.linalg.norm(res)**2 + lambda_val * np.sum(np.abs(x))
            self.loss_history.append(loss)
            
            if np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-12) < self.tol:
                break
        return x

    def huber_nesterov(self, lambda_ratio=0.05, alpha=1.0, mu=1e-2):
        """加速 Huber 梯度下降 (Nesterov)"""
        x = self._AT_op(self.y)
        y_k = x.copy()
        t_k = 1.0
        lambda_val = lambda_ratio * np.max(np.abs(x))
        self.loss_history = []
        
        for i in range(self.max_iter):
            x_prev = x.copy()
            
            # 在 y_k (外推点) 处计算梯度
            Ay = self._A_op(y_k)
            res = Ay - self.y
            grad_data = self._AT_op(res)
            grad_reg = lambda_val * self._grad_huber(y_k, mu)
            
            # 梯度更新
            x = y_k - alpha * (grad_data + grad_reg)
            
            # Nesterov 动量更新
            t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            beta = (t_k - 1) / t_next
            y_k = x + beta * (x - x_prev)
            t_k = t_next
            
            loss = 0.5 * np.linalg.norm(res)**2 + lambda_val * np.sum(np.abs(x))
            self.loss_history.append(loss)
            
            if np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-12) < self.tol:
                break
        return x

# ==========================================
# 1. 主渲染函数
# ==========================================
def render():
    # --- 全局配置 ---
    st.markdown("""
    <style>
    .main-text {font-size:16px; line-height:1.6;}
    .math-box {background-color:#eef9fe; padding:15px; border-radius:10px; border-left: 5px solid #2196f3;}
    </style>
    """, unsafe_allow_html=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    tab_theory, tab_sim, tab_analysis = st.tabs(["📖 平滑化原理", "🔬 ISAR 仿真实验室", "📉 收敛性能诊断"])

    # ==========================================
    # Tab 1: 理论深度解析
    # ==========================================
    with tab_theory:
        st.markdown("### 1. 为什么需要平滑？(Smoothing)")
        st.markdown("""
        <div class="main-text">
        在 LASSO 问题中，在零点处是<b>不可导</b>的。
        这导致我们无法直接使用标准的梯度下降法（Gradient Descent），而必须使用次梯度法或近端梯度法（如 ISTA）。
        <br><br>
        <b>Huber 平滑</b>提供了一种替代思路：用一个光滑的函数 $H_\mu(x)$ 来逼近 $|x|$。
        这样，目标函数就变得处处可导，我们可以直接使用高效的梯度类算法（甚至 L-BFGS 等高级优化器）。
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"""
        H_\mu(x) = \begin{cases} 
        \frac{1}{2\mu} x^2 & \text{if } |x| \le \mu \quad (\text{二次函数，平滑}) \\
        |x| - \frac{\mu}{2} & \text{if } |x| > \mu \quad (\text{线性函数，稀疏})
        \end{cases}
        """)

        st.info("""
        **参数 $\mu$ (Mu) 的作用**：
        *   **$\mu$ 很大**：函数接近 $x^2$ (Ridge 回归)，容易优化，但稀疏性差。
        *   **$\mu$ 很小**：函数接近 $|x|$ (Lasso)，稀疏性好，但梯度变化剧烈，优化困难。
        """)

        # --- 交互演示: Huber 函数 ---
        st.markdown("#### 🧪 实验：Huber 函数形态")
        
        col_demo_ctrl, col_demo_viz = st.columns([1, 2])
        with col_demo_ctrl:
            mu_demo = st.slider("平滑参数 (Mu)", 0.01, 2.0, 0.5, 0.01)
            st.caption("观察原点附近的圆滑程度。")
        
        with col_demo_viz:
            x = np.linspace(-3, 3, 500)
            l1 = np.abs(x)
            huber = np.where(np.abs(x) <= mu_demo, 0.5 * x**2 / mu_demo, np.abs(x) - 0.5 * mu_demo)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(x, l1, 'k--', alpha=0.3, label='L1 Norm |x|')
            ax.plot(x, huber, 'b-', linewidth=2, label=f'Huber (mu={mu_demo})')
            
            # 放大原点细节
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(0, 1.5)
            ax.set_title("L1 vs Huber Approximation")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ==========================================
    # Tab 2: 仿真运行
    # ==========================================
    with tab_sim:
        st.markdown("### 🚀 Huber 梯度法实战")
        
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 参数设置")
            
            with st.expander("📡 采样设置", expanded=True):
                sampling_rate = st.slider("降采样率 (SR)", 0.1, 0.8, 0.4, 0.05)
            
            with st.expander("🧠 算法设置", expanded=True):
                algo_type = st.radio("优化策略", ["Huber Gradient Descent", "Huber Accelerated (Nesterov)"])
                lambda_ratio = st.slider("正则化强度 (Lambda)", 0.01, 0.20, 0.05, 0.01)
                mu_val = st.number_input("平滑参数 (Mu)", value=0.01, format="%.4f", step=0.005)
                alpha = st.slider("步长 (Alpha)", 0.1, 2.0, 1.0, 0.1)
                max_iter = st.number_input("最大迭代次数", 20, 500, 100, 20)
            
            run_btn = st.button("开始重建", type="primary")

        # --- 数据准备 (修正版) ---
        # 1. 获取原始数据
        raw_orig, _, _, _ = get_isar_data()
        
        # 2. 归一化 (关键步骤：Huber 参数 mu 对数据尺度敏感)
        raw = raw_orig / np.max(np.abs(raw_orig))
        
        # 3. 基于归一化数据计算 Ground Truth
        rd_img = compute_rd_image(raw)
        
        N_pulses = raw.shape[1]
        
        # 随机采样掩模
        np.random.seed(42) 
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :] 
        
        # 传统 RD 结果（基准）
        img_fft = compute_rd_image(raw_sparse)
        # 翻转以符合视觉习惯
        img_fft = np.flipud(img_fft) 
        rd_img_disp = np.flipud(rd_img)

        with col_main:
            if run_btn:
                progress_bar = st.progress(0)
                status = st.empty()
                
                # 初始化求解器
                solver = HuberISARSolver(raw_sparse, mask, max_iter=max_iter)
                
                t_start = time.time()
                
                # --- 运行算法 ---
                if algo_type == "Huber Gradient Descent":
                    recon = solver.huber_gradient_descent(lambda_ratio, alpha, mu_val)
                else:
                    recon = solver.huber_nesterov(lambda_ratio, alpha, mu_val)
                
                t_end = time.time()
                duration = t_end - t_start
                recon = np.flipud(recon)
                
                progress_bar.progress(100)
                status.success(f"✅ 计算完成！耗时: {duration:.3f} 秒")
                
                # 绘图对比
                fig_res = plot_isar_comparison(rd_img_disp, img_fft, recon, algo_type, duration, sampling_rate)
                st.pyplot(fig_res)
                
                # 保存状态供分析
                st.session_state['huber_run'] = {
                    'loss_history': solver.loss_history,
                    'params': {'lambda': lambda_ratio, 'mu': mu_val, 'algo': algo_type}
                }
                
                # 结果解读
                st.info(f"""
                **分析**：
                Huber 方法通过平滑化 L1 范数，使得目标函数变得可微。
                *   如果结果不够稀疏（背景噪声大），尝试**减小 Mu** 或 **增大 Lambda**。
                *   如果收敛太慢，尝试使用 **Nesterov 加速**。
                """)

            else:
                st.info("👈 请调整左侧参数并运行仿真。")
                # 显示占位图
                fig_holder = plot_isar_comparison(rd_img_disp, img_fft, np.zeros_like(rd_img), "Waiting...", 0, sampling_rate)
                st.pyplot(fig_holder)

    # ==========================================
    # Tab 3: 收敛分析
    # ==========================================
    with tab_analysis:
        st.markdown("### 📈 算法收敛性诊断")
        
        if 'huber_run' in st.session_state:
            run_data = st.session_state['huber_run']
            loss_hist = run_data['loss_history']
            
            if len(loss_hist) > 0:
                col_an1, col_an2 = st.columns([2, 1])
                
                with col_an1:
                    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                    ax_loss.plot(loss_hist, 'b-o', markersize=3, linewidth=1.5, label='Huber Loss')
                    ax_loss.set_title("Objective Function Descent")
                    ax_loss.set_xlabel("Iteration (k)")
                    ax_loss.set_ylabel("Loss Value")
                    ax_loss.grid(True, linestyle='--', alpha=0.5)
                    ax_loss.legend()
                    st.pyplot(fig_loss)
                    
                with col_an2:
                    st.markdown("#### 诊断报告")
                    init_loss = loss_hist[0]
                    final_loss = loss_hist[-1]
                    
                    st.metric("初始 Loss", f"{init_loss:.2e}")
                    st.metric("最终 Loss", f"{final_loss:.2e}")
                    
                    st.markdown(f"""
                    **当前设置**：
                    * 算法: `{run_data['params']['algo']}`
                    * Mu: `{run_data['params']['mu']}`
                    
                    **观察**：
                    Huber 方法通常比 ISTA 收敛更平滑，因为它避免了硬性的阈值截断操作。
                    """)
            else:
                st.warning("无 Loss 数据。")
        else:
            st.info("请先在 [仿真实验] 标签页运行一次算法。")

if __name__ == "__main__":
    render()