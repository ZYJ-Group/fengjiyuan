import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import styler, play_animation 

def render():
    # --- 1. 全局配置：解决 Matplotlib 中文和负号显示问题 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    # 分层讲解
    tab_def, tab_anim = st.tabs(["📐 严谨定义与公式解析", "📺 动态演示"])

    # --- Tab 1: 严谨定义 (保持不变) ---
    with tab_def:
        col_def, col_viz = st.columns([1, 1.6])
        with col_def:
            st.markdown(r"""
            ### 1. 什么是次梯度？
            对于凸函数 $f$，向量 $g$ 是在点 $x$ 处的一个**次梯度**，如果它满足核心不等式：
            
            $$ \underbrace{f(y)}_{\text{真实值}} \geq \underbrace{f(x) + g^T(y-x)}_{\text{线性预测值}}, \quad \forall y $$
            
            > **公式深度解析** (看右图)：
            > * 我们站在固定的点 **$x$** (黑点)。
            > * 我们选定一个斜率/方向 **$g$**。
            > * 考察任意另一个点 **$y$**。
            > * **不等式的含义**：函数在 $y$ 点的真实高度（蓝点），永远要高于或等于基于 $g$ 做出的线性预测高度（绿点）。
            
            ### 2. 全局线性下界
            这个公式本质上说明，由 $g$ 确定的线性函数（超平面）是原函数的一个**全局支撑**，它永远托在函数图像的下方，绝不会刺破它。

            ### 3. 什么是次微分？
            在点 $x$ 处**所有**合法次梯度的集合，称为**次微分 (Subdifferential)**，记为 $\partial f(x)$。
            
            * **光滑点**：集合里只有一个元素（梯度）。
            * **尖点 (Kink)**：集合里有无穷多个元素（一个区间）。
            """)
        
        with col_viz:
            fig_formula, ax = styler.get_fig_ax("可视化：次梯度公式深度解析")
            y_vals = np.linspace(-2.5, 2.5, 200)
            f_vals = np.abs(y_vals)
            x_fixed = 0          
            f_x_fixed = 0
            g_val = 0.5          
            line_vals = f_x_fixed + g_val * (y_vals - x_fixed)

            ax.plot(y_vals, f_vals, 'b-', linewidth=3, label=r'原函数 $f(y)=|y|$', zorder=5)
            ax.plot(y_vals, line_vals, color='#28a745', linestyle='--', linewidth=2, label=r'线性下界 $f(x) + g(y-x)$')
            
            y_point = 1.8
            fy_point = np.abs(y_point)
            ly_point = f_x_fixed + g_val * (y_point - x_fixed)

            ax.scatter([x_fixed], [f_x_fixed], c='black', s=100, zorder=10, label=r'固定点 $x$')
            ax.vlines(y_point, -0.5, fy_point, color='gray', linestyle=':', linewidth=1.5)
            ax.text(y_point, -0.7, r'任意点 $y$', ha='center', fontsize=11, fontweight='bold')
            ax.scatter([y_point], [fy_point], c='blue', s=80, zorder=10)
            ax.scatter([y_point], [ly_point], c='#28a745', s=80, zorder=10)

            ax.annotate(r'$f(y)$', xy=(y_point, fy_point), xytext=(y_point - 1.5, fy_point + 0.2),
                        arrowprops=dict(facecolor='blue', edgecolor='none', shrink=0.05, width=3),
                        fontsize=12, color='blue', fontweight='bold')

            ax.annotate(r'$f(x) + g^T(y-x)$', xy=(y_point, ly_point), xytext=(y_point + 0.5, ly_point - 0.5),
                        arrowprops=dict(facecolor='#28a745', edgecolor='none', shrink=0.05, width=3),
                        fontsize=12, color='#28a745', fontweight='bold', ha='left')

            ax.annotate('', xy=(y_point, ly_point), xytext=(y_point, fy_point),
                        arrowprops=dict(arrowstyle='<->', color='#D32F2F', linewidth=2))
            ax.text(y_point + 0.1, (fy_point + ly_point) / 2, r'$\geq$', color='#D32F2F', fontsize=14, va='center', fontweight='bold')

            ax.set_ylim(-1, 3)
            ax.set_xlim(-2.5, 2.5)
            ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
            st.pyplot(fig_formula)

    # --- Tab 2: 动态演示 (保持不变) ---
    with tab_anim:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📺 动态扫描演示")
            st.markdown(r"""
            **场景**：检测 $f(x)=|x|$ 在 $x=0$ 处的次梯度。
            
            **动画说明**：旋转直线的斜率 $g$。
            1.  **✅ 绿色区间** ($g \in [-1, 1]$)：直线完全在函数下方，是合法的次梯度。
            2.  **❌ 红色区间** (其他)：斜率太陡，直线刺破函数图像，违例。
            """)
            run_anim = st.button("▶️ 播放：次梯度扫描", key="btn_subg")

        with col2:
            def draw_subg_frame(g):
                is_valid = -1 <= g <= 1
                if is_valid:
                    color = '#28a745'
                    status_text = r"✅ 合法: $g \in [-1, 1]$"
                else:
                    color = '#dc3545'
                    status_text = r"❌ 违例: 刺破函数图像"
                
                title = f"斜率 g={g:.2f} | {status_text}"
                fig, ax = styler.get_fig_ax(title)
                x = np.linspace(-2.5, 2.5, 300)
                f_y = np.abs(x)
                line_y = g * x
                
                ax.plot(x, f_y, 'b-', linewidth=3, label=r'$f(x)=|x|$', zorder=5)
                ax.plot(x, line_y, color=color, linestyle='--', linewidth=2, label=r'测试直线 $g^T(y-x)$')
                
                if not is_valid:
                    ax.fill_between(x, f_y, line_y, where=(line_y > f_y), color='red', alpha=0.3, label='违反区域')
                else:
                    ax.fill_between(x, -3, line_y, color='green', alpha=0.1)

                ax.scatter([0], [0], c='black', s=60, zorder=10)
                ax.set_ylim(-2, 2.5)
                ax.set_xlim(-2, 2)
                ax.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white')
                return fig

            if run_anim:
                gs = np.concatenate([np.linspace(-2, -1.1, 8), np.linspace(-1, 1, 15), np.linspace(1.1, 2, 8)])
                play_animation(draw_subg_frame, gs, interval=0.1)
            else:
                st.pyplot(draw_subg_frame(0.5))