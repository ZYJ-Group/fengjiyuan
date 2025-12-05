import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- 1. 样式与工具配置 (保持不变) ---
class Styler:
    def __init__(self):
        # 尝试兼容更多中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Source Han Sans CN', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False 

    def get_fig_ax(self, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, linestyle='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
        return fig, ax

styler = Styler()

def play_animation_in_streamlit(draw_func, frames, interval=0.1):
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    total_frames = len(frames)
    for i, val in enumerate(frames):
        fig = draw_func(val)
        frame_placeholder.pyplot(fig)
        plt.close(fig)
        progress_bar.progress((i + 1) / total_frames)
        time.sleep(interval)
    progress_bar.empty()

# --- 2. 绘图逻辑函数库 (保持不变) ---

def draw_conjugate_geometry(slope_y):
    x_opt = slope_y
    f_val = 0.5 * x_opt**2
    linear_val = slope_y * x_opt
    f_star = linear_val - f_val 
    
    title = f"斜率 y={slope_y:.2f} | 共轭值 $f^*(y)={f_star:.2f}$"
    fig, ax = styler.get_fig_ax(title)
    
    x = np.linspace(-1, 4.5, 200)
    f_x = 0.5 * x**2
    linear_func = slope_y * x
    
    ax.plot(x, f_x, color='#1f77b4', linewidth=2, label=r'$f(x) = \frac{1}{2}x^2$')
    ax.plot(x, linear_func, color='#ff7f0e', linestyle='--', alpha=0.6, label=r'线性项 $y^T x$')
    ax.vlines(x_opt, f_val, linear_val, color='#d62728', linewidth=3, zorder=10, label='Max Gap')
    
    tangent_line = slope_y * x - f_star
    ax.plot(x, tangent_line, color='purple', linestyle='-.', linewidth=1.5, label='支撑超平面')
    
    intercept = -f_star
    ax.plot([0, 0], [0, intercept], color='purple', linestyle=':', alpha=0.5)
    ax.scatter([0], [intercept], color='purple', s=80, marker='D', zorder=10)
    ax.text(0.1, intercept, f"截距 = $-f^*(y)$", color='purple', fontsize=9, va='center')

    ax.set_ylim(-5, 9) 
    ax.set_xlim(-1, 4.5)
    ax.legend(loc='upper left')
    return fig

def draw_lagrange_2d(lam):
    # 问题: min x^2  s.t. x=2
    # L(x, lam) = x^2 + lam*(x-2)
    # 最优 x 由 dL/dx = 2x + lam = 0 决定 -> x = -lam/2
    x_opt_current = -lam / 2
    g_lambda = x_opt_current**2 + lam * (x_opt_current - 2) 
    
    fig, ax = styler.get_fig_ax(r"拉格朗日函数 $L(x, \lambda) = x^2 + \lambda(x-2)$")
    
    x = np.linspace(-4, 6, 200)
    f_x = x**2
    L_x = x**2 + lam * (x - 2)
    
    ax.plot(x, f_x, 'k--', label=r'原目标 $f(x)=x^2$', alpha=0.3)
    ax.plot(x, L_x, '#1f77b4', linewidth=2.5, label=r'当前 $L(x, \lambda)$')
    
    ax.scatter([x_opt_current], [g_lambda], color='red', s=100, zorder=5, label=r'$\min_x L$ (即 $g(\lambda)$)')
    ax.vlines(x_opt_current, -15, g_lambda, linestyles=':', color='red', alpha=0.5)
    
    ax.axvline(2, color='green', linewidth=2, alpha=0.5, label='约束 x=2')
    ax.scatter([2], [4], color='green', marker='*', s=200, zorder=10, label='原问题最优解 p*=4')

    ax.set_ylim(-10, 15)
    ax.set_xlim(-4, 6)
    ax.legend(loc='upper left')
    
    return fig, g_lambda

def draw_saddle_3d():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x_3d = np.linspace(0, 4, 30)
    lam_3d = np.linspace(-6, -2, 30) 
    X, LAM = np.meshgrid(x_3d, lam_3d)
    Z = X**2 + LAM * (X - 2) 
    ax.plot_surface(X, LAM, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.scatter([2], [-4], [4], color='red', s=100, label='Saddle Point')
    ax.set_title("Lagrange Function Surface")
    ax.set_xlabel('x')
    ax.set_ylabel('lambda')
    ax.set_zlabel('L')
    ax.view_init(elev=25, azim=130)
    return fig

# --- 3. 主页面渲染 ---
def render():
    tab_def, tab_duality, tab_anim = st.tabs(["📐 基础：共轭定义", "⚖️ 核心：从原问题到对偶", "📺 演示：共轭几何"])

    # === Tab 1: 基础定义 ===
    with tab_def:
        col_text, col_img = st.columns([1, 1.3])
        with col_text:
            st.markdown(r"""
            ### 1. 什么是共轭函数？
            $$ f^*(y) = \sup_{x} \{ y^T x - f(x) \} $$
            
            **直观理解 (Max Gap)**：
            给定斜率 $y^T$，我们在 $f(x)$ 上找一点，使得线性函数 $y^Tx$ 与 $f(x)$ 的**垂直距离最大**。
            
            ### 2. 核心性质
            *   **永远是凸的**：无论 $f(x)$ 长什么样，$f^*(y)$ 都是凸函数（是一族仿射函数的逐点上确界）。
            *   **Legendre 变换**：如果 $f(x)$ 是光滑凸函数，最大值点处满足 $\nabla f(x) = y$。
            """)
        with col_img:
            fig_sup, ax_sup = styler.get_fig_ax("可视化证明：直线的上确界必为凸")
            y_range = np.linspace(-3, 3, 300)
            x_samples = np.linspace(-2.5, 2.5, 15) 
            supremum_y = np.full_like(y_range, -np.inf)
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(x_samples)))
            for i, x_val in enumerate(x_samples):
                intercept = -0.5 * x_val**2
                z_vals = x_val * y_range + intercept
                supremum_y = np.maximum(supremum_y, z_vals)
                ax_sup.plot(y_range, z_vals, color=colors[i], alpha=0.3)
            ax_sup.plot(y_range, supremum_y, color='#D32F2F', linestyle='--', linewidth=3, label=r'$f^*(y)$')
            ax_sup.legend()
            st.pyplot(fig_sup)

# === Tab 2: 进阶对偶理论 (整合了详细推导) ===
    with tab_duality:
        st.markdown("### 为什么需要对偶？")
        
        # 使用更精准的数学描述替代原来的模糊表述
        st.info("""
        💡 **将一个原始问题 (Primal)转化为**对偶问题 (Dual)**，是因为对偶问题拥有两个不可替代的数学优势：
        
        1.  **去约束化 (Unconstrained)**：
            原问题必须在可行域内搜索（可行域限制 $Ax=b$）；对偶问题则将其转化为目标函数中的“惩罚项” $\lambda^T(Ax-b)$，从而可以在全空间自由搜索。
            
        2.  **必为凸优化 (Convexity)**：
            这是对偶理论最强大的性质——**无论原函数 $f(x)$ 是什么样子，对偶函数 $g(\lambda)$ 永远是凹函数（Concave）！** 这意味着求解对偶问题永远是一个容易的凸优化问题。
        """)

        # 步骤 2：对偶函数与共轭的联系
        st.markdown("#### 第二步：寻找下界与共轭的联系 (The Missing Link)")
        
        # --- 整合：详细推导部分 ---
        with st.expander("🔗 点击查看数学推导：为什么 g(λ) 实际上是 f*(y)？", expanded=True):
            st.markdown(r"""
            我们想求对偶函数 $g(\lambda) = \inf_x L(x, \lambda)$。
            这个过程其实就是把“求极小值问题”转化为“求共轭函数值”的过程。
            
            **推导分三步走：**

            **第一步：整理项与提取常数**
            拉格朗日函数为 $L(x, \lambda) = f(x) + \lambda^T (Ax - b)$。
            我们将与 $x$ 无关的项移到 $\inf$ 外面，并利用向量内积性质 $\lambda^T Ax = (A^T \lambda)^T x$：
            
            $$
            \begin{aligned}
             g(\lambda) &= \inf_x \left( f(x) + \lambda^T Ax - \lambda^T b \right) \\
                        &= \inf_x \left( f(x) + (A^T \lambda)^T x \right) \underbrace{- \lambda^T b}_{\text{常数项}}
            \end{aligned}
            $$

            **第二步：利用负号转换 (inf $\to$ sup)**
            共轭函数的定义是关于 $\sup$ (最大值) 的，而这里是 $\inf$ (最小值)。
            我们需要利用恒等式 $\inf (h(x)) = - \sup (-h(x))$ 进行翻转：
            
            $$
            \begin{aligned}
             g(\lambda) &= - \sup_x \left( - \left[ f(x) + (A^T \lambda)^T x \right] \right) - \lambda^T b \\
                        &= - \sup_x \left( \color{blue}{- (A^T \lambda)^T x - f(x)} \right) - \lambda^T b
            \end{aligned}
            $$

            **第三步：模式匹配 (配凑共轭定义)**
            回顾共轭函数的标准定义： $\color{#d62728}{f^*(y) = \sup_x ( y^T x - f(x) )}$。
            
            仔细观察上面 $\sup$ 括号里的蓝色部分：
            * 这里的 $x$ 对应定义里的 $x$。
            * 这里的 $-f(x)$ 对应定义里的 $-f(x)$。
            * **关键点**：这里的 $\color{blue}{- (A^T \lambda)}$ 正好占据了定义中 $\color{#d62728}{y}$ 的位置。
            
            $$
            \begin{aligned}
             g(\lambda) &= - \underbrace{\sup_x \left( (\color{#d62728}{-A^T \lambda})^T x - f(x) \right)}_{f^*(-A^T \lambda)} - \lambda^T b \\
                        &= \boxed{- f^*(-A^T \lambda) - b^T \lambda}
            \end{aligned}
            $$
            
            ---
            **结论**：
            求对偶函数 $g(\lambda)$ 不需要重新做优化，只需要**查表**找到 $f$ 的共轭函数 $f^*$，然后把变量 $-A^T\lambda$ 代进去即可！
            """)
        # ---------------------

        col_deriv_1, col_deriv_2 = st.columns(2)
        with col_deriv_1:
            st.success(r"""
            **验证例子：** $\min x^2 \quad \text{s.t. } x=2$
            
            **方法 A：直接求导法**
            1. $L = x^2 + \lambda(x-2)$
            2. $dL/dx = 2x+\lambda=0 \Rightarrow x = -\lambda/2$
            3. 代入：$(-\lambda/2)^2 + \lambda(-\lambda/2 - 2)$
            4. 结果：$\color{blue}{-\frac{\lambda^2}{4} - 2\lambda}$
            """)
        with col_deriv_2:
            st.success(r"""
            **方法 B：共轭公式法**
            1. $f(x)=x^2 \Rightarrow f^*(y) = y^2/4$
            2. 这里 $A=1, b=2$
            3. 公式：$-f^*(-A^T\lambda) - b^T\lambda$
            4. 代入：$-f^*(-\lambda) - 2\lambda = -(-\lambda)^2/4 - 2\lambda$
            5. 结果：$\color{blue}{-\frac{\lambda^2}{4} - 2\lambda}$
            
            **结果完全一致！** 🎯
            """)

        st.divider()

        # 步骤 3：对偶问题与可视化
        st.markdown("#### 第三步：对偶问题 (寻找最好的下界)")
        st.markdown(r"""
        对偶问题就是最大化这个下界： $\max_\lambda g(\lambda)$。
        """)

        # 交互演示
        st.subheader("交互：观察下界如何逼近最优解")
        col_ctrl, col_viz = st.columns([1, 2])
        with col_ctrl:
            st.markdown(r"""
            **实验说明**：
            * 拖动 $\lambda$，寻找能让 $g(\lambda)$ 最大的点。
            * 注意：对于凸问题，最大化的 $g(\lambda)$ 将正好等于原问题的最小值 $p^*$。
            """)
            lam_val = st.slider(r"调整乘子 $\lambda$", -6.0, 2.0, 0.0, 0.2)
            
            # 计算并显示当前状态
            g_val = -(lam_val**2)/4 - 2*lam_val
            gap = 4.0 - g_val
            st.metric("当前下界 g(λ)", f"{g_val:.2f}", delta=f"距离 p* 还有: {gap:.2f}", delta_color="inverse")
            
            if abs(gap) < 0.05:
                st.balloons()
                st.success(r"🎉 恭喜！你找到了 $\lambda^* = -4$，此时 $g(\lambda^*) = p^*$！")

        with col_viz:
            fig_lag, _ = draw_lagrange_2d(lam_val)
            st.pyplot(fig_lag)

        # 步骤 4：鞍点视角
        with st.expander("👁️ 进阶视角：为什么叫“鞍点”(Saddle Point)？"):
            col_saddle_text, col_saddle_img = st.columns([1, 1.5])
            with col_saddle_text:
                st.markdown(r"""
                如果你把 $x$ 和 $\lambda$ 同时看作变量，最优解 $(x^*, \lambda^*)$ 就在马鞍面的中心：
                
                $$ \min_x \max_\lambda L(x, \lambda) $$
                
                1. **对 $x$ (原变量)**：它是谷底（最小化成本）。
                2. **对 $\lambda$ (对偶变量)**：它是山峰（最大化下界）。
                """)
            with col_saddle_img:
                st.pyplot(draw_saddle_3d())

    # === Tab 3: 共轭几何演示 (保持不变) ===
    with tab_anim:
        col_ctrl, col_plot = st.columns([1, 2.5])
        with col_ctrl:
            st.markdown("""
            **共轭几何演示**
            
            观察重点：
            1. **红色 Gap**：$y^Tx - f(x)$ 的最大值。
            2. **紫色切线**：当 Gap 最大时，切线截距正好是 $-f^*(y)$。
            """)
            mode = st.radio("演示模式", ["🖱️ 手动交互", "▶️ 自动播放"], horizontal=True)
            
            if mode == "🖱️ 手动交互":
                slope_val = st.slider("斜率 y", 0.0, 3.0, 1.5)
            else:
                run_btn = st.button("▶️ 播放动画")
                slope_val = 1.5

        with col_plot:
            if mode == "🖱️ 手动交互":
                st.pyplot(draw_conjugate_geometry(slope_val))
            elif run_btn:
                slopes = np.linspace(0.0, 3.0, 50)
                play_animation_in_streamlit(draw_conjugate_geometry, slopes)
            else:
                st.pyplot(draw_conjugate_geometry(1.5))

if __name__ == "__main__":
    render()