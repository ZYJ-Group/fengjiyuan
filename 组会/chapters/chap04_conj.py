import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- 1. 样式与工具配置 (保持不变) ---
class Styler:
    def __init__(self):
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
            给定斜率 $y$，我们在 $f(x)$ 上找一点，使得线性函数 $y^Tx$ 与 $f(x)$ 的**垂直距离最大**。
            
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

    # === Tab 2: 进阶对偶理论 (重点修改部分) ===
    with tab_duality:
        st.markdown("### 为什么需要对偶？")
        st.info("💡 我们的目标：把一个“有硬性约束”的困难问题，转化为一个“无约束”的简单问题。")

        # 步骤 1：构建拉格朗日函数
        st.markdown("#### 第一步：软化约束 (罚函数思想)")
        st.markdown(r"""
        原问题：$\min f(x) \quad \text{s.t. } Ax = b$
        
        如果直接求解很难，我们可以引入一个**价格（乘子 $\lambda$）**来把约束变成目标函数的一部分：
        $$ L(x, \lambda) = f(x) + \lambda^T (Ax - b) $$
        *   $f(x)$：原本的成本。
        *   $\lambda^T (Ax - b)$：违反规则的**罚款**。
        """)

        # 步骤 2：对偶函数（下界性质）
        st.markdown("#### 第二步：寻找下界 (Lower Bound)")
        col_deriv_1, col_deriv_2 = st.columns(2)
        with col_deriv_1:
            st.markdown(r"""
            定义**对偶函数** $g(\lambda)$ 为 $L(x, \lambda)$ 关于 $x$ 的最小值：
            $$ g(\lambda) = \inf_x L(x, \lambda) $$
            
            **关键性质**：对于任意满足约束的可行解 $\tilde{x}$ (即 $A\tilde{x}=b$)：
            $$ L(\tilde{x}, \lambda) = f(\tilde{x}) + \lambda^T(0) = f(\tilde{x}) $$
            因为 $g(\lambda)$ 是 $L$ 的最小值，所以自然有：
            $$ \color{red}{g(\lambda) \le f(\tilde{x})} $$
            
            **结论：$g(\lambda)$ 永远是原问题最优解 $p^*$ 的下界！**
            """)
        with col_deriv_2:
            st.success(r"""
            **举例：** $\min x^2 \quad \text{s.t. } x=2$
            
            1. $L(x, \lambda) = x^2 + \lambda(x-2)$
            2. 对 $x$ 求导求极小：$2x + \lambda = 0 \Rightarrow x = -\lambda/2$
            3. 代回得到对偶函数：
               $g(\lambda) = (-\frac{\lambda}{2})^2 + \lambda(-\frac{\lambda}{2} - 2) = -\frac{\lambda^2}{4} - 2\lambda$
               
            这是一个开口向下的抛物线，它的最大值就是原问题的下界。
            """)

        st.divider()

        # 步骤 3：对偶问题与可视化
        st.markdown("#### 第三步：对偶问题 (寻找最好的下界)")
        st.markdown(r"""
        既然 $g(\lambda)$ 是下界，我们当然希望这个下界**越紧越好**（越大越好）。
        所以对偶问题就是：
        $$ \max_\lambda g(\lambda) $$
        """)

        # 交互演示
        st.subheader("🧪 交互实验：观察下界如何逼近最优解")
        col_ctrl, col_viz = st.columns([1, 2])
        with col_ctrl:
            st.markdown(r"""
            **实验说明**：
            *   原问题：$\min x^2, \text{s.t. } x=2$。
            *   最优解（绿色星）：$p^* = 4$。
            *   红色曲线：$L(x, \lambda)$。
            *   红色点：$L$ 的最低点，即 $g(\lambda)$。
            
            **任务**：拖动 $\lambda$，试着让红色点（下界）尽可能接近绿色星（最优解）。
            """)
            lam_val = st.slider(r"调整乘子 $\lambda$", -6.0, 2.0, 0.0, 0.2)
            
            # 计算并显示当前状态
            g_val = -(lam_val**2)/4 - 2*lam_val
            gap = 4.0 - g_val
            st.metric("当前下界 g(λ)", f"{g_val:.2f}", delta=f"距离 p* 还有: {gap:.2f}", delta_color="inverse")
            
            if abs(gap) < 0.05:
                st.balloons()
                st.success(r"🎉 恭喜！你找到了 $\lambda^* = -4$，此时 $g(\lambda^*) = p^*$，强对偶成立！")

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
    st.set_page_config(page_title="共轭与对偶完全解析", layout="wide")
    render()