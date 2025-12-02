import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

# 仅保留必要的工具函数引用
from utils import styler, play_animation 

def render():
    st.markdown("## 📏 向量范数 (Vector Norms)")

    # 使用标签页分层展示内容
    tab1, tab2 = st.tabs(["📐 数学定义", "📺 动态演示"])

    # --- Tab 1: 数学定义 ---
    with tab1:
        st.markdown(r"""
        ### 1. 什么是范数？
        范数是将向量映射到非负实数的函数 $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$，它赋予了向量空间“长度”的概念。
        
        一个严格的范数必须满足以下**三个条件**：
        1.  **非负性与定性**：$\|x\| \ge 0$，且仅当 $x=0$ 时 $\|x\|=0$。
        2.  **齐次性**：$\|\alpha x\| = |\alpha| \|x\|$（缩放向量，长度成比例缩放）。
        3.  **三角不等式**：$\|x+y\| \le \|x\| + \|y\|$（两点之间直线最短）。

        ### 2. $L_p$ 范数家族
        通项公式：$\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$
        
        | 范数 | 常用名称 | 公式 (二维) | 几何形状 | 特性 |
        | :--- | :--- | :--- | :--- | :--- |
        | **$L_0$** | (伪范数) | 非零元素个数 | 十字架 | 理想的稀疏性，但难以优化 (NP-Hard) |
        | **$L_1$** | 曼哈顿距离 | $\vert x_1 \vert + \vert x_2 \vert$ | **菱形** | **稀疏解** (Lasso)，$L_0$ 的最优凸近似 |
        | **$L_2$** | 欧几里得距离 | $\sqrt{x_1^2 + x_2^2}$ | **圆形** | **平滑解** (Ridge)，各向同性 |
        | **$L_\infty$** | 切比雪夫距离 | $\max(\vert x_1 \vert, \vert x_2 \vert)$ | **正方形** | 关注最大误差，鲁棒控制 |
        """)
        
        st.info("💡 **注**：当 $0 < p < 1$ 时，不再满足三角不等式，单位球是**内凹**的 (Non-convex)，严格来说称为“准范数” (Quasi-norm)。")

    # --- Tab 2: 核心用途 & 几何直觉 ---
    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("范数 $L_p$ 球体形状演变")
            st.markdown(r"""
            通过调整 $p$ 值，观察单位球 $\{x : \|x\|_p \le 1\}$ 的形状变化：

            *   **$0 < p < 1$**：内凹 (Non-convex)，如星形。
            *   **$p = 1$**：**菱形** (L1)，稀疏性的临界点。
            *   **$p = 2$**：**圆形** (L2)，最自然的欧氏距离。
            *   **$p \to \infty$**：**正方形** (L-inf)，边界向外扩张。
            """)
            run_anim = st.button("▶️ 播放形状演变动画", key="btn_norm")

        with col2:
            def draw_norm_frame(p_val):
                # 动态标题与状态判断
                if p_val < 0.99:
                    status_text = "Quasi-norm (Non-convex)"
                    color = "#FFA500" # Orange for warning
                elif abs(p_val - 1.0) < 0.05:
                    status_text = "L1 Norm (Diamond)"
                    color = "#FF4B4B" # Red
                elif abs(p_val - 2.0) < 0.05:
                    status_text = "L2 Norm (Circle)"
                    color = "#0000FF" # Blue
                elif p_val > 5.0:
                    status_text = "L-inf Norm (Square)"
                    color = "#000000"
                else:
                    status_text = f"L{p_val:.1f} Norm"
                    color = "#333333"

                title_str = r"$\|x\|_p = 1, \quad p = " + f"{p_val:.2f}$"
                
                # 使用 utils 中的 styler 获取统一风格画布
                fig, ax = styler.get_fig_ax(title_str)
                
                # 网格数据
                x = np.linspace(-1.5, 1.5, 250)
                y = np.linspace(-1.5, 1.5, 250)
                X, Y = np.meshgrid(x, y)
                
                # 计算范数 Z = (|x|^p + |y|^p)^(1/p)
                # 加上 1e-6 防止底数为0
                Z = (np.abs(X)**p_val + np.abs(Y)**p_val)**(1/p_val)
                
                # 绘制
                ax.contourf(X, Y, Z, levels=[0, 1], colors=[color], alpha=0.3)
                ax.contour(X, Y, Z, levels=[1], colors=[color], linewidths=2.5)
                
                # 辅助框
                rect = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='gray', linestyle='--', alpha=0.4)
                ax.add_patch(rect)
                
                # 标注状态
                ax.text(0, 1.65, status_text, ha='center', fontsize=12, color=color, fontweight='bold')
                
                ax.set_xlim(-1.6, 1.6)
                ax.set_ylim(-1.6, 1.8) # 留出顶部写字空间
                ax.set_aspect('equal')
                return fig

            if run_anim:
                # 构造 p 值序列：
                # 1. 从 0.5 到 0.9 (Non-convex)
                # 2. 在 1.0 处停顿 (重复多次)
                # 3. 从 1.1 到 1.9
                # 4. 在 2.0 处停顿
                # 5. 快速增加到 8.0
                p_values = np.concatenate([
                    np.linspace(0.5, 0.9, 8),
                    [1.0] * 8,                  # Pause at L1
                    np.linspace(1.1, 1.9, 10),
                    [2.0] * 8,                  # Pause at L2
                    np.linspace(2.2, 8.0, 15)
                ])
                play_animation(draw_norm_frame, p_values, interval=0.08)
            else:
                # 默认显示 L1
                st.pyplot(draw_norm_frame(1.0))