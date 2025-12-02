import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.optimize import linprog
# 替换旧版 fftpack 为新的 fft 接口
from scipy.fft import dct, idct

# ==========================================
# 0. 基础配置与工具类 (Utils)
# ==========================================

st.set_page_config(page_title="压缩感知演示", layout="centered")

# --- 字体配置 (增强版) ---
def configure_fonts():
    system_name = platform.system()
    fonts = []
    if system_name == "Windows":
        fonts = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    elif system_name == "Darwin":  # Mac
        fonts = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
    else:  # Linux / Streamlit Cloud
        # 增加更多 Linux 常见中文字体候补
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
    
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False

configure_fonts()

# --- 样式辅助类 ---
class Styler:
    def get_fig_ax(self, title="", figsize=(6, 4.5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax

styler = Styler()

# ==========================================
# 1. 核心算法函数 (Algorithms)
# ==========================================

# --- 2D DCT 变换 (使用 scipy.fft) ---
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# --- L1 最小化求解器 (封装 Linprog) ---
def solve_bp(A, y, n):
    """
    求解基追踪 (Basis Pursuit): min ||x||_1 s.t. Ax = y
    转化为线性规划:
    变量 z = [x, t], 长度 2n
    min sum(t)
    s.t. Ax = y
         -t <= x <= t  =>  x - t <= 0  &  -x - t <= 0
    """
    m = A.shape[0]
    
    # 1. 目标函数 c: 前n个为0(对应x), 后n个为1(对应t)
    c_lp = np.concatenate([np.zeros(n), np.ones(n)])
    
    # 2. 等式约束 A_eq * z = b_eq
    # [A, 0] * [x, t]^T = y
    A_eq = np.hstack([A, np.zeros((m, n))])
    b_eq = y
    
    # 3. 不等式约束 A_ub * z <= b_ub
    # [ I, -I] * [x, t]^T <= 0
    # [-I, -I] * [x, t]^T <= 0
    eye = np.eye(n)
    A_ub = np.vstack([
        np.hstack([eye, -eye]),
        np.hstack([-eye, -eye])
    ])
    b_ub = np.zeros(2 * n)
    
    # 4. 变量边界
    # x: (-inf, inf), t: [0, inf)
    bounds = [(None, None)] * n + [(0, None)] * n
    
    # 5. 求解
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                  bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:n]
    else:
        return np.zeros(n)

# --- ISTA 求解器 (优化版) ---
def ista_solve(mask, observed, lambda_val=0.01, max_iter=100, tol=1e-5):
    """
    使用迭代软阈值算法 (ISTA) 求解图像恢复
    增加了收敛检测 (tol) 以提高效率
    """
    n_r, n_c = mask.shape
    x = np.zeros((n_r, n_c))
    alpha = 0.9  # 学习率 (对于正交基+Mask采样，谱半径接近1，0.9较安全且快)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(max_iter):
        x_prev = x.copy()
        
        # 1. 重建图像
        img_curr = idct2(x)
        
        # 2. 计算残差 (mask * img - observed)
        diff = (mask * img_curr) - observed
        
        # 3. 梯度更新 (转回频域)
        grad_freq = dct2(diff)
        x_temp = x - alpha * grad_freq
        
        # 4. 软阈值 (Soft Thresholding)
        x = np.sign(x_temp) * np.maximum(np.abs(x_temp) - lambda_val, 0)
        
        # 5. 收敛检测
        if i > 0 and i % 5 == 0:
            change = np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-8)
            if change < tol:
                status_text.caption(f"✅ 收敛于迭代 {i}")
                progress_bar.progress(1.0)
                break

        if i % 10 == 0:
            progress_bar.progress((i + 1) / max_iter)
            status_text.caption(f"ISTA 优化中... Iter {i}/{max_iter}")
            
    progress_bar.empty()
    status_text.empty()
    return idct2(x)

# ==========================================
# 2. 主渲染函数 (Render)
# ==========================================

def render():

# --- Part 1: 核心定义与理论体系 ---
    st.markdown("### 📚 1. 核心定义与颠覆性思想")
    
    # 1.1 直观对比：传统 vs CS
    st.markdown("#### 1.1 采样哲学的根本变革")
    col_def1, col_def2 = st.columns([1, 1], gap="medium")
    
    with col_def1:
        st.error("👴 **传统理论 (Nyquist-Shannon)**")
        st.markdown("""
        *   **核心法则**：采样频率必须 $\ge 2 \times f_{max}$。
        *   **工作流**：
            1.  **全采样**：先获取海量数据 (如 4000万像素)。
            2.  **压缩**：抛弃 90% 的系数 (如存为 JPEG)。
            3.  **存储/传输**。
        *   **痛点**：对于高频信号（MRI、高速视频），传感器成本极高，且浪费大量算力在“先采后扔”上。
        """)

    with col_def2:
        st.success("🚀 **压缩感知 (Compressed Sensing)**")
        st.markdown("""
        *   **核心法则**：采样数 $M$ 仅需略大于信号的**稀疏度** $K$。
        *   **工作流**：
            1.  **压缩采样**：通过随机投影，直接获取 $M$ 个观测值 ($M \ll N$)。
            2.  **传输**：数据量极小。
            3.  **重构**：利用算法计算出原始信号。
        *   **优势**：大幅降低了对传感器带宽和存储的需求。
        """)

    # 1.2 数学基石
    st.markdown("#### 1.2 为什么这能实现？(三大基石)")
    
    with st.expander("💡 点击展开：深入理解数学原理与重建逻辑", expanded=True):
        st.markdown("""
        压缩感知的成功依赖于三个关键数学条件：
        """)
        
        # 基石 1: 稀疏性
        st.markdown("**1️⃣ 稀疏性 (Sparsity)**")
        st.latex(r"x = \Psi s, \quad \|s\|_0 \le K \ll N")
        st.caption("""
        自然界的信号（图像、声音）在时域可能看起来很乱，但在某个变换域（如 $\Psi$ = DCT、小波变换）下，
        **大部分系数接近为零**，只有 $K$ 个大系数包含了主要信息。这是 CS 能工作的前提。
        """)
        
        st.divider()
        
        # 基石 2: 观测矩阵与 RIP
        st.markdown("**2️⃣ 不相干观测 (Incoherence & RIP)**")
        st.latex(r"y = \Phi x = \Phi \Psi s")
        st.caption("""
        我们需要设计一个观测矩阵 $\Phi$（如高斯随机矩阵），它不能“破坏”信号的稀疏结构。
        这要求 $\Phi$ 满足 **有限等距性质 (RIP)**：
        确保稀疏向量变换后的长度基本不变，从而保证信息的唯一性。
        """)
        
        st.divider()
        
        # 基石 3: 优化重构
        st.markdown("**3️⃣ 非线性重建 (L1 Minimization)**")
        st.info("我们需要解欠定方程组 $y = Ax$。因为方程少、未知数多，解有无穷多个。我们需要找到**最稀疏**的那个解。")
        
        cols_math = st.columns(3)
        with cols_math[0]:
            st.markdown("❌ **$L_2$ (最小二乘)**")
            st.latex(r"\min \|x\|_2 \text{ s.t. } Ax=y")
            st.markdown("解是平滑的，**不稀疏**，无法恢复原信号。")
        with cols_math[1]:
            st.markdown("❌ **$L_0$ (理想)**")
            st.latex(r"\min \|x\|_0 \text{ s.t. } Ax=y")
            st.markdown("直接数非零个数。**NP-hard 难解问题**，计算量爆炸。")
        with cols_math[2]:
            st.markdown("✅ **$L_1$ (凸松弛)**")
            st.latex(r"\min \|x\|_1 \text{ s.t. } Ax=y")
            st.markdown("**CS 的魔法**：$L_1$ 既能诱导稀疏解，又是凸优化问题(可解)。")

    st.divider()

    # --- Part 2: 交互实验 ---
    tab_sim, tab_img = st.tabs([
        "🎹 1D 信号恢复",
        "🖼️ 2D 图像修复" 
    ])
    # --- Tab 2: 1D 信号 ---
    with tab_sim:
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("N (长度)", 50, 200, 100)
        k = c2.number_input("K (稀疏度)", 1, 20, 5)
        m = c3.slider("M (观测数)", 10, n, 30)
        
        if st.button("🚀 运行 1D 恢复", key="btn_1d"):
            # 1. 生成数据
            # 固定随机种子确保结果可复现（可选）
            # np.random.seed(42) 
            x_true = np.zeros(n)
            indices = np.random.choice(n, k, replace=False)
            x_true[indices] = np.random.choice([-1, 1], k) * (np.random.rand(k) * 2 + 2) # 避免太接近0
            
            A = np.random.randn(m, n)
            y = A @ x_true
            
            # 2. 求解
            with st.spinner("求解 L1 线性规划中..."):
                x_l1 = solve_bp(A, y, n)
                x_l2 = np.linalg.pinv(A) @ y # L2 最小二乘解
            
            # 3. 绘图
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
            # L2 Plot
            axes[0].set_title("L2 最小二乘 (失败): 噪声充斥全频段")
            axes[0].stem(range(n), x_true, linefmt='k-', markerfmt='ko', basefmt=" ", label='真值')
            axes[0].plot(range(n), x_l2, 'g-', alpha=0.6, label='L2 恢复')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            
            # L1 Plot
            err = np.linalg.norm(x_true - x_l1)
            status = "完美" if err < 1e-3 else "有误差"
            color = 'green' if err < 1e-3 else 'red'
            
            axes[1].set_title(f"L1 压缩感知 ({status}): 误差 {err:.2e}", color=color, fontweight='bold')
            axes[1].stem(range(n), x_true, linefmt='k-', markerfmt='ko', basefmt=" ", label='真值')
            axes[1].plot(range(n), x_l1, 'r.', markersize=8, label='L1 恢复')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- Tab 3: 2D 图像 ---
    with tab_img:
        col_ctrl, col_view = st.columns([1, 2])
        
        with col_ctrl:
            st.write("#### 参数设置")
            sample_rate = st.slider("采样率 (保留像素)", 0.1, 0.8, 0.4, 0.05)
            # 使用 session_state 防止每次滑动滑块都重算
            if 'run_img' not in st.session_state:
                st.session_state.run_img = False
            
            if st.button("✨ 开始修复图像"):
                st.session_state.run_img = True
            
            st.info(f"观测矩阵大小: {sample_rate*100:.1f}% 的像素被保留。\n\n利用 DCT 域的稀疏性进行重建。")

        with col_view:
            if st.session_state.run_img:
                # 1. 制造图像
                size = 64
                img_orig = np.zeros((size, size))
                # 简单的几何图形 + 梯度
                xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
                img_orig[xx**2 + yy**2 < 0.5] = 0.8
                img_orig[np.abs(xx - yy) < 0.2] = 0.5
                img_orig += 0.2 * xx # 添加渐变使其在DCT域不完全是0，增加难度
                
                # 2. 采样
                mask = np.random.rand(size, size) < sample_rate
                img_corrupted = img_orig * mask
                
                # 3. 求解
                # 动态调整 lambda，采样越少 lambda 应该稍微大一点以去噪
                rec_img = ista_solve(mask, img_corrupted, lambda_val=0.01, max_iter=200)
                
                # 4. 显示
                fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                
                # Helper to plot
                def plot_im(ax, img, t):
                    im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(t)
                    ax.axis('off')
                    return im

                plot_im(axes[0], img_orig, "原始图像")
                plot_im(axes[1], img_corrupted, f"采样输入 ({sample_rate:.0%})")
                plot_im(axes[2], rec_img, "CS 恢复结果")
                
                st.pyplot(fig)
                plt.close(fig)
                
                # PSNR
                mse = np.mean((img_orig - rec_img)**2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                st.success(f"图像恢复完成! PSNR: {psnr:.2f} dB")
            else:
                st.info("👈 请点击左侧按钮运行仿真")

if __name__ == "__main__":
    render()