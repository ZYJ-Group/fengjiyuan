import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, PathPatch, FancyArrowPatch
from matplotlib.path import Path
from scipy.spatial import ConvexHull, QhullError
import platform

# 1. 设置后端
matplotlib.use("Agg") 

# --- 🛠️ 绘图工具类 ---
class PlotUtils:
    def __init__(self):
        system_name = platform.system()
        if system_name == "Windows":
            self.font_list = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        elif system_name == "Darwin": 
            self.font_list = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
        else:
            self.font_list = ["DejaVu Sans", "sans-serif"]
        
        plt.rcParams['font.sans-serif'] = self.font_list
        plt.rcParams['axes.unicode_minus'] = False 

    def get_canvas(self, title):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(title, fontsize=12, color='#333333', pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, alpha=0.2)
        ax.axvline(0, color='black', linewidth=1, alpha=0.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig, ax

utils = PlotUtils()

# --- 🎨 核心绘图逻辑 ---

def plot_convex_definition(theta):
    """绘制凸集定义"""
    fig, ax = utils.get_canvas(r"凸集判定: $\theta x + (1-\theta)y \in C$")
    
    pts = np.array([[-1, -0.5], [-0.5, 1], [0.5, 1], [1, -0.5], [0, -1.2]])
    poly = Polygon(pts, closed=True, color='#2A9D8F', alpha=0.5, label='凸集 $C$')
    ax.add_patch(poly)
    
    p1 = np.array([-0.8, -0.2]) 
    p2 = np.array([0.7, 0.2])   
    p_curr = (1 - theta) * p1 + theta * p2
    
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5, linewidth=1.5)
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='black', s=40, zorder=5)
    ax.text(p1[0]-0.2, p1[1], "$x$", fontsize=12)
    ax.text(p2[0]+0.1, p2[1], "$y$", fontsize=12)
    ax.scatter([p_curr[0]], [p_curr[1]], c='#E76F51', s=100, zorder=10, edgecolors='white', label=r'动点 $\theta$')
    
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    return fig

def plot_non_convex(theta):
    """绘制非凸集"""
    fig, ax = utils.get_canvas("非凸集：连线跑出集合外部")
    
    verts = [(-1, 1), (-1, -1), (1, -1), (1, 1), (0.5, 1), (0.5, -0.5), (-0.5, -0.5), (-0.5, 1)]
    verts_for_path = verts + [(0, 0)] 
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1) + [Path.CLOSEPOLY]
    
    path = Path(verts_for_path, codes)
    patch = PathPatch(path, facecolor='#E63946', alpha=0.5, lw=0)
    ax.add_patch(patch)
    
    p1 = np.array([-0.8, 0.8])
    p2 = np.array([0.8, 0.8])
    p_curr = (1 - theta) * p1 + theta * p2
    
    is_inside = path.contains_point(p_curr)
    status_color = '#2A9D8F' if is_inside else '#E63946'
    status_text = "IN (内部)" if is_inside else "OUT (外部!)"
    
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5)
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='black', s=30)
    ax.scatter([p_curr[0]], [p_curr[1]], c=status_color, s=100, zorder=10, edgecolors='white')
    ax.set_title(f"状态: {status_text}", color=status_color, fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    return fig

def plot_affine_demo(shear, scale, rotate_deg):
    """
    绘制仿射变换保凸性演示
    原理: f(x) = Ax + b
    """
    fig, ax = utils.get_canvas(r"仿射变换: $C \to f(C)$")
    
    # 1. 定义原始凸集（正方形）
    original_pts = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    
    # 2. 构造变换矩阵 A
    # 旋转
    theta = np.radians(rotate_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta),  np.cos(theta)]])
    # 剪切 (Shear) 和 缩放 (Scale)
    S = np.array([[scale, shear], 
                  [0,     scale]])
    
    # 组合矩阵 A = R * S
    A = R @ S
    
    # 3. 计算变换后的点
    # 注意矩阵乘法顺序，pts是(N,2)，所以用 pts @ A.T
    transformed_pts = original_pts @ A.T 
    
    # 4. 绘图
    # 绘制原始集合（虚线）
    poly_orig = Polygon(original_pts, closed=True, fill=False, linestyle='--', edgecolor='gray', linewidth=1.5, label='原始凸集 $C$')
    ax.add_patch(poly_orig)
    
    # 绘制变换后的集合（实色）
    poly_trans = Polygon(transformed_pts, closed=True, color='#457B9D', alpha=0.6, label=r'变换后 $f(C)$')
    ax.add_patch(poly_trans)
    
    # 添加一个箭头指示变换
    ax.annotate("", xy=(transformed_pts[0,0], transformed_pts[0,1]), xytext=(original_pts[0,0], original_pts[0,1]),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.5))

    ax.text(0, -1.8, "线性/仿射变换保持凸性\n正方形 $\\to$ 平行四边形", ha='center', fontsize=10, color='#555')

    # 动态调整坐标轴以适应变形
    limit = max(1.5, np.max(np.abs(transformed_pts))) + 0.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.legend(loc='upper right')
    
    return fig

# --- 🖥️ 主渲染程序 ---
def render():
    st.markdown(r"""
    > **定义**：集合 $C$ 是凸集，当且仅当对于任意 $x, y \in C$ 和 $\theta \in [0, 1]$，都有：
    > $$ \theta x + (1-\theta)y \in C $$
    """)

    tab1, tab2, tab3 = st.tabs([
        "🟢 几何定义", 
        "🔶 常见凸集", 
        "❌ 非凸反例"
    ])

    # === Tab 1: 定义演示 ===
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("👇 拖动滑块，观察点是否始终在图形内部。")
            theta = st.slider("参数 theta", 0.0, 1.0, 0.5, 0.05)
            st.latex(r"p = (1-\theta)x + \theta y")
        with col2:
            fig = plot_convex_definition(theta)
            st.pyplot(fig)
            plt.close(fig)

    # === Tab 2: 常见凸集 ===
    with tab2:
        st.markdown("#### 书中常见的凸集类型")
        
        # 增加了一个新选项：仿射变换
        shape_type = st.radio(
            "选择演示类型", 
            ["范数球 (Norm Balls)", "多面体 (Polyhedra)", "仿射函数 (Affine Maps)"], 
            horizontal=True
        )

        col_a, col_b = st.columns([1, 1.5])
        
        if shape_type == "范数球 (Norm Balls)":
            with col_a:
                st.markdown(r"""
                **$L_p$ 范数球**：$\{x \mid \|x\|_p \le 1\}$
                
                * $L_1$ (绿色): 菱形
                * $L_2$ (蓝色): 圆形
                * $L_\infty$ (橙色): 正方形
                """)
            with col_b:
                fig, ax = utils.get_canvas("范数球对比")
                ax.add_patch(Rectangle((-1, -1), 2, 2, color='#F4A261', alpha=0.3, label=r'$L_\infty$'))
                ax.add_patch(Circle((0, 0), 1, color='#457B9D', alpha=0.5, label=r'$L_2$'))
                diamond = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
                ax.add_patch(Polygon(diamond, color='#2A9D8F', alpha=0.6, label=r'$L_1$'))
                ax.legend()
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                st.pyplot(fig)
                plt.close(fig)

        elif shape_type == "多面体 (Polyhedra)":
            with col_a:
                st.markdown(r"""
                **多面体**：$P = \{x \mid Ax \le b\}$
                
                是有限个半空间和超平面的交集。
                图示展示了4个半空间的交集区域。
                """)
            with col_b:
                fig, ax = utils.get_canvas("多面体: 半空间的交集")
                verts = np.array([[-0.6, -1.0], [ 0.8, -0.6], [ 0.6,  0.8], [-0.8,  0.4]])
                poly = Polygon(verts, closed=True, color='#264653', alpha=0.7, label='交集区域')
                ax.add_patch(poly)
                # 简单绘制延伸线
                ax.plot([-2, 2], [-1.2, -0.4], 'k--', alpha=0.3) 
                ax.plot([0.8, 0.4], [-2, 2], 'k--', alpha=0.3)
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                st.pyplot(fig)
                plt.close(fig)
        
        elif shape_type == "仿射函数 (Affine Maps)":
            with col_a:
                st.markdown(r"""
                **仿射变换保持凸性**
                
                令 $f(x) = Ax + b$ 为仿射函数。
                * 如果 $C$ 是凸集，则其像 $f(C)$ 也是凸集。
                * 如果 $D$ 是凸集，则其原像 $f^{-1}(D)$ 也是凸集。
                
                **调节参数观察变换：**
                """)
                shear = st.slider("剪切 (Shear)", -1.0, 1.0, 0.5, step=0.1)
                scale = st.slider("缩放 (Scale)", 0.5, 2.0, 1.0, step=0.1)
                rotate = st.slider("旋转 (Rotation)", 0, 90, 15)
                
            with col_b:
                # 调用新的绘图函数
                fig = plot_affine_demo(shear, scale, rotate)
                st.pyplot(fig)
                plt.close(fig)

    # === Tab 3: 非凸集 ===
    with tab3:
        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.error("❌ 非凸集特征")
            st.markdown("存在“凹陷”部分。连接集合内两点的线段，可能会经过集合外部。")
            t_val = st.slider("移动连线检测", 0.0, 1.0, 0.5, 0.05, key="slider_non_cvx")
        
        with col_d:
            fig = plot_non_convex(t_val)
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="凸集可视化", layout="wide")
    render()