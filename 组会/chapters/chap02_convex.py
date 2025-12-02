import streamlit as st
import numpy as np
import matplotlib
# 1. 设置非交互式后端
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError
import time
import platform

# --- 🛠️ 内置工具函数 ---
class Styler:
    def __init__(self):
        # 2. 自动选择中文字体
        system_name = platform.system()
        if system_name == "Windows":
            self.font_name = "SimHei" # Windows 黑体
        elif system_name == "Darwin":
            self.font_name = "Arial Unicode MS" # Mac 通用
        else:
            self.font_name = "DejaVu Sans" # Linux/其他默认
            
        plt.rcParams['font.sans-serif'] = [self.font_name]
        plt.rcParams['axes.unicode_minus'] = False 

    def get_fig_ax(self, title):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax.set_aspect('equal')
        return fig, ax

def play_animation(draw_func, frames, interval=0.1):
    """
    动画播放器
    """
    frame_placeholder = st.empty()
    for f in frames:
        fig = draw_func(f)
        frame_placeholder.pyplot(fig, use_container_width=False)
        plt.close(fig) 
    
    # 最后一帧
    fig = draw_func(frames[-1])
    frame_placeholder.pyplot(fig, use_container_width=False)
    plt.close(fig)

styler = Styler()

# --- 🖥️ 主渲染函数 ---
def render():

    # --- 1. 理论部分 ---
    with st.expander("📘 凸集理论摘要", expanded=False):
        st.markdown(r"""
        **1. 定义对比**：
        * **仿射集 (Affine Set)**：过集合内任意两点的**直线**都在集合内。
        * **凸集 (Convex Set)**：连接集合内任意两点的**线段**都在集合内 。
            * 公式：$\theta x + (1-\theta)y \in C, \quad \forall \theta \in [0,1]$。
        * **凸锥 (Convex Cone)**：集合是凸的，且对任意 $x \in C, \lambda \ge 0 \Rightarrow \lambda x \in C$。

        **2. 重要的凸集**：
        * **超平面与半空间**：$\{x \mid a^Tx \le b\}$。
        * **范数球**：$\{x \mid \|x\|_p \le r\}$。
        * **多面体 (Polyhedra)**：有限个半空间和超平面的交集。
        """)

    # --- 2. 演示部分 ---
    tab_concept, tab_important_sets, tab_hull, tab_non_convex = st.tabs([
        "🟢 凸集判定演示", 
        "🔶 重要的凸集展示", 
        "📦 凸包 (Convex Hull)",
        "❌ 非凸集演示"
    ])

    # === Tab 1: 凸集判定 ===
    with tab_concept:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("判定标准：集合内任意两点连线是否完全在集合内部？")
            run_convex = st.button("▶️ 播放：凸集连线测试", key="btn_cvx")
        
        with col2:
            def draw_convex_frame(theta):
                fig, ax = styler.get_fig_ax(f"凸集判定: theta={theta:.2f}")
                # 凸多边形
                pts = np.array([[-1, -0.5], [-0.5, 1], [0.5, 1], [1, -0.5], [0, -1.2]])
                poly = Polygon(pts, closed=True, color='#66b3ff', alpha=0.6, label='凸集 C')
                ax.add_patch(poly)
                
                # 动点
                p1 = np.array([-0.8, -0.2])
                p2 = np.array([0.7, 0.2])
                p_curr = (1 - theta) * p1 + theta * p2
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.4)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='black', s=30)
                ax.scatter([p_curr[0]], [p_curr[1]], c='#28a745', s=100, zorder=10, edgecolors='white', label='动点')
                
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                ax.legend(loc='upper right')
                return fig  # 返回单个 fig 对象

            if run_convex:
                thetas = np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
                play_animation(draw_convex_frame, thetas, interval=0.05)
            else:
                # 【修复】去掉 [0]，因为 draw_convex_frame 现在只返回 fig
                st.pyplot(draw_convex_frame(0.5), use_container_width=False)
                plt.close()

    # === Tab 2: 重要的凸集 ===
    with tab_important_sets:
        st.markdown("##### 基于书中 2.4.2 节“重要的凸集”")
        set_type = st.radio(
            "选择凸集类型:",
            ["范数球 (Norm Balls)", "多面体 (Polyhedra)", "凸锥 (Convex Cone)"],
            horizontal=True
        )

        if set_type == "范数球 (Norm Balls)":
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(r"""**范数球**：$C = \{x \mid \|x\|_p \le 1\}$""")
            with col_b:
                fig, ax = styler.get_fig_ax("不同范数定义的单位球")
                ax.add_patch(Rectangle((-1, -1), 2, 2, color='#ffcc99', alpha=0.4, label=r'$l_\infty$'))
                ax.add_patch(Circle((0, 0), 1, color='#66b3ff', alpha=0.6, label=r'$l_2$'))
                diamond_pts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
                ax.add_patch(Polygon(diamond_pts, closed=True, color='#28a745', alpha=0.7, label=r'$l_1$'))
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                ax.legend(loc='upper right', fontsize='small')
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        elif set_type == "多面体 (Polyhedra)":
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(r"""**多面体**：有限个半空间和超平面的交集。""")
            with col_b:
                fig, ax = styler.get_fig_ax("多面体")
                x = np.linspace(-2, 2, 200)
                y1 = np.ones_like(x)
                y2 = -np.ones_like(x)
                y3 = 1.5 - x
                y4 = 1.5 + x
                y_bottom = np.maximum(y2, -2)
                y_top = np.minimum(np.minimum(y1, y3), y4)
                ax.fill_between(x, y_bottom, y_top, where=(y_top >= y_bottom), color='purple', alpha=0.5)
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        elif set_type == "凸锥 (Convex Cone)":
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(r"""**二阶锥**：$y \ge |x|$""")
            with col_b:
                fig, ax = styler.get_fig_ax("凸锥")
                x = np.linspace(-1.5, 1.5, 200)
                y1 = np.abs(x)       
                y2 = np.full_like(x, 2.0)
                ax.fill_between(x, y1, y2, color='orange', alpha=0.4)
                ax.plot(x, y1, 'k-', lw=2, alpha=0.6)
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.2, 1.5)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

    # === Tab 3: 凸包 ===
    with tab_hull:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown(r"""**凸包**：包含点集的最小凸集。""")
            gen_points = st.button("🎲 生成随机点集")
            
        with col_b:
            if 'hull_points' not in st.session_state or gen_points:
                st.session_state.hull_points = np.random.rand(15, 2) * 2 - 1 
            
            points = st.session_state.hull_points
            fig, ax = styler.get_fig_ax("凸包演示")
            ax.scatter(points[:,0], points[:,1], c='blue')
            
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    vertices_idx = hull.vertices
                    plot_idx = np.concatenate([vertices_idx, [vertices_idx[0]]])
                    ax.plot(points[plot_idx, 0], points[plot_idx, 1], 'r-', lw=2)
                    ax.fill(points[vertices_idx, 0], points[vertices_idx, 1], 'red', alpha=0.2)
                except QhullError:
                    st.warning("⚠️ 点集共线")
            
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    # === Tab 4: 非凸集 ===
    with tab_non_convex:
        col3, col4 = st.columns([1, 2])
        with col3:
            st.error("非凸集：存在“凹陷”，连线会跑出集合。")
            run_non_convex = st.button("▶️ 播放：非凸测试", key="btn_non_cvx")
            
        with col4:
            def draw_non_convex_frame(theta):
                fig, ax = styler.get_fig_ax(f"非凸集测试: theta={theta:.2f}")
                verts = [(-1, -1), (-1, 1), (1, 1), (0, 0), (1, -1)]
                path_obj = MplPath(verts + [(-1, -1)])
                
                poly = Polygon(verts, closed=True, color='#ff9999', alpha=0.6, label='非凸集')
                ax.add_patch(poly)
                
                p1 = np.array([0.6, 0.8])
                p2 = np.array([0.6, -0.8])
                p_curr = (1 - theta) * p1 + theta * p2
                
                is_inside = path_obj.contains_point(p_curr)
                color = '#28a745' if is_inside else '#dc3545'
                status = "IN" if is_inside else "OUT"
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.4)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='black', s=30)
                ax.scatter([p_curr[0]], [p_curr[1]], c=color, s=100, zorder=10, edgecolors='white')
                ax.text(p_curr[0]+0.1, p_curr[1], status, color=color, fontweight='bold')
                
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                ax.legend(loc='upper right')
                return fig # 返回单个 fig 对象
                
            if run_non_convex:
                thetas = np.concatenate([np.linspace(0, 1, 30), np.linspace(1, 0, 30)])
                play_animation(draw_non_convex_frame, thetas, interval=0.06)
            else:
                # 【修复】关键修改：去掉 [0]，直接传入返回的 fig
                st.pyplot(draw_non_convex_frame(0.5), use_container_width=False)
                plt.close()

if __name__ == "__main__":
    render()