import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import time

# --- 1. 将辅助函数移出 render，避免重复定义 ---

def simulate_gradient_descent(func_type, start_x, learning_rate=0.05, steps=50):
    """
    模拟小球滚动的轨迹
    """
    path_x = [start_x]
    path_y = []
    
    current_x = start_x
    
    for _ in range(steps):
        # 1. 计算当前高度 y
        if func_type == 'non_convex':
            # y = 0.5 * (x - 1.5)^2 - 2 * sin(2*x)
            y = 0.5 * (current_x - 1.5)**2 - 2 * np.sin(2 * current_x)
            # 导数: y' = (x - 1.5) - 4 * cos(2*x)
            grad = (current_x - 1.5) - 4 * np.cos(2 * current_x)
        else: # convex
            # y = (x - 1)^2
            y = (current_x - 1)**2
            # 导数: y' = 2 * (x - 1)
            grad = 2 * (current_x - 1)
        
        path_y.append(y)
        
        # 2. 梯度下降更新 x (模拟重力滚落)
        current_x = current_x - learning_rate * grad
        path_x.append(current_x)
    
    # 计算最后一个点的y值
    if func_type == 'non_convex':
        path_y.append(0.5 * (path_x[-1] - 1.5)**2 - 2 * np.sin(2 * path_x[-1]))
    else:
        path_y.append((path_x[-1] - 1)**2)
        
    return path_x, path_y

def create_animated_plot(title, x_range, y_func_type, path_x, path_y, annotations=[]):
    """
    创建 Plotly 动画
    """
    # 1. 背景曲线数据
    x_curve = np.linspace(x_range[0], x_range[1], 400)
    
    # --- 修复 Bug: 逻辑判断修改 ---
    if y_func_type == 'non_convex': 
        y_curve = 0.5 * (x_curve - 1.5)**2 - 2 * np.sin(2 * x_curve)
    else:
        y_curve = (x_curve - 1)**2

    # 2. 创建基础图形
    fig = go.Figure(
        data=[
            # 背景曲线
            go.Scatter(x=x_curve, y=y_curve, mode="lines", name="函数曲线", line=dict(color='royalblue', width=2)),
            # 起始点（小球）
            go.Scatter(x=[path_x[0]], y=[path_y[0]], mode="markers", name="优化算法(小球)",
                       marker=dict(color="red", size=15, symbol="circle"))
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(range=[x_range[0], x_range[1]], title="参数 X"),
            yaxis=dict(range=[np.min(y_curve)-1, np.max(y_curve)+1], title="损失 Loss"),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="▶️ 开始下降",
                              method="animate",
                              args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])]
            )]
        )
    )

    # 3. 创建动画帧 (Frames)
    frames = []
    for i in range(len(path_x)):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_curve, y=y_curve, mode="lines", line=dict(color='royalblue', width=2)), # 保持背景
                go.Scatter(x=[path_x[i]], y=[path_y[i]], mode="markers", 
                           marker=dict(color="red", size=15)) # 移动小球
            ]
        ))
    fig.frames = frames

    # 4. 添加标注 (Annotations)
    for ann in annotations:
        fig.add_annotation(x=ann['x'], y=ann['y'], text=ann['text'], 
                           showarrow=True, arrowhead=1, ax=ann['ax'], ay=ann['ay'])

    return fig

# --- 2. 替代缺失的 utils 功能 ---
def get_fig_ax(title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig, ax

def run_matplotlib_animation(plot_func, param_values, placeholder):
    """简单的 Matplotlib 动画播放器"""
    for val in param_values:
        fig = plot_func(val)
        placeholder.pyplot(fig)
        time.sleep(0.05)
        plt.close(fig) # 防止内存泄漏

# --- 主渲染函数 ---
def render():
    # 修复 Matplotlib 中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    # 选项卡
    tab_def, tab_1st, tab_2nd, tab_opt = st.tabs(["📐 0阶: 几何定义 (Jensen)", "📉 1阶: 切线条件", "🚀 2阶: 海森矩阵", "🔍 优化性质"])

    # --- Tab 1: 几何定义 (Jensen) ---
    with tab_def:
        st.markdown(r"""
        ### 0阶定义：弦在弧之上
        凸函数最直观的定义是：函数图像上任意两点之间的连线（弦），都在函数图像的**上方**。
        
        **詹森不等式 (Jensen's Inequality)**：
        $$ f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y), \quad \forall \theta \in [0,1] $$
        """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("📺 **动画演示**\n\n观察红点（弦上的点）是否始终高于绿点（函数上的点）。")
            run_jensen = st.button("▶️ 播放：詹森不等式演示", key="btn_jensen")
        
        with col2:
            placeholder_jensen = st.empty() # 用于动画占位
            
            def draw_jensen_frame(theta):
                # 替换了 styler.get_fig_ax
                fig, ax = get_fig_ax(f"Jensen 不等式: theta={theta:.2f}")
                
                # 函数 f(x) = x^2
                x_base = np.linspace(-1.5, 2.5, 200)
                ax.plot(x_base, x_base**2, 'b-', linewidth=2, alpha=0.6, label=r'$f(x)=x^2$')
                
                # 选定两点 A, B
                xA, xB = -1.0, 2.0
                yA, yB = xA**2, xB**2
                
                # 1. 画弦
                ax.plot([xA, xB], [yA, yB], 'k--', linewidth=1.5, label='弦')
                ax.scatter([xA, xB], [yA, yB], c='black', s=40)
                
                # 2. 计算点
                x_interp = theta * xA + (1 - theta) * xB
                y_chord = theta * yA + (1 - theta) * yB
                y_func = x_interp**2
                
                # 3. 绘制对比
                ax.vlines(x_interp, y_func, y_chord, color='gray', linestyle=':', linewidth=2)
                ax.scatter([x_interp], [y_chord], c='red', s=80, zorder=10, label='弦上点')
                ax.scatter([x_interp], [y_func], c='green', s=80, zorder=10, label='函数点')
                
                ax.set_ylim(-1, 5)
                ax.set_xlim(-1.5, 2.5)
                ax.legend(loc='upper center')
                return fig

            if run_jensen:
                thetas = np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
                run_matplotlib_animation(draw_jensen_frame, thetas, placeholder_jensen)
            else:
                placeholder_jensen.pyplot(draw_jensen_frame(0.5))

    # --- Tab 2: 一阶条件 (切线) ---
    with tab_1st:
        st.markdown(r"""
        ### 1阶条件：全局线性下界
        如果 $f$ 可微，凸函数图像始终位于其**任意切线**（一阶泰勒近似）的**上方**。
        
        $$ f(y) \geq f(x) + \nabla f(x)^T(y-x) $$
        
        这意味着切线是函数的**全局线性下界**。
        
        """)
        
        col3, col4 = st.columns([1, 2])
        with col3:
            st.info("📺 **动画演示**\n\n红色虚线（切线）永远托住蓝色函数曲线。")
            run_tangent = st.button("▶️ 播放：切线扫描演示", key="btn_tangent")
            
        with col4:
            placeholder_tangent = st.empty()
            
            def draw_tangent_frame(x0):
                fig, ax = get_fig_ax(f"一阶条件: 切点 x0={x0:.2f}")
                x = np.linspace(-2.2, 2.2, 200)
                y = x**2 
                
                y0 = x0**2
                slope = 2*x0
                tangent = slope * (x - x0) + y0
                
                ax.plot(x, y, 'b-', linewidth=3, label=r'$f(x)$')
                ax.plot(x, tangent, 'r--', linewidth=2, label='切线')
                ax.scatter([x0], [y0], c='red', zorder=10, s=60)
                ax.fill_between(x, y, tangent, color='green', alpha=0.15)
                
                ax.set_ylim(-4, 5)
                ax.set_xlim(-2.2, 2.2)
                ax.legend(loc='upper center')
                return fig

            if run_tangent:
                x_vals = np.linspace(-2, 2, 30)
                run_matplotlib_animation(draw_tangent_frame, x_vals, placeholder_tangent)
            else:
                placeholder_tangent.pyplot(draw_tangent_frame(1.0))

    # --- Tab 3: 二阶条件 (曲面) ---
    with tab_2nd:
        st.markdown(r"""
        ### 2阶条件：海森矩阵半正定
        若 $f$ 二阶可微，则 $f$ 是凸函数当且仅当其**海森矩阵**是**半正定**的即： 
                    
        ($\nabla^2 f(x) \succeq 0$)。
        """)
        
        col5, col6 = st.columns([1, 2])
        with col5:
            st.info("""
            **图示说明**
            右图展示了二元凸函数 $f(x, y) = x^2 + y^2$。
            形状像一个碗，曲率在任何方向都是向上的。
            """)
            
        with col6:
            X = np.arange(-2, 2.1, 0.1)
            Y = np.arange(-2, 2.1, 0.1)
            X, Y = np.meshgrid(X, Y)
            Z = X**2 + Y**2  

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, edgecolor='none')
            ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.viridis)
            ax.set_title(r'凸函数曲面: $f(x,y) = x^2 + y^2$')
            st.pyplot(fig)

    # --- Tab 4: 优化性质 (Plotly 动画) ---
    with tab_opt:
        st.subheader("凸函数的特殊性")
        st.markdown("凸优化问题有一个极其重要的性质：**“任何局部最优解都是全局最优解。”**")

        col_non, col_cvx = st.columns(2)

        # --- 左图：非凸函数 ---
        with col_non:
            st.markdown("##### ❌ 非凸函数 (Non-Convex)")
            # 模拟数据
            path_x, path_y = simulate_gradient_descent('non_convex', start_x=4.2, learning_rate=0.04, steps=60)
            
            anns = [
                {'x': 3.7, 'y': -0.8, 'text': '局部最优(陷阱)', 'ax': 0, 'ay': -40},
                {'x': 0.9, 'y': -1.9, 'text': '全局最优', 'ax': 0, 'ay': 40}
            ]
            
            # 注意这里传入 'non_convex' 字符串，修复了之前的 Bug
            fig1 = create_animated_plot(
                title="容易卡在局部最优",
                x_range=[-1.5, 4.5],
                y_func_type="non_convex", 
                path_x=path_x,
                path_y=path_y,
                annotations=anns
            )
            st.plotly_chart(fig1, use_container_width=True)

        # --- 右图：凸函数 ---
        with col_cvx:
            st.markdown("##### ✅ 凸函数 (Convex)")
            # 模拟数据
            path_x_cvx, path_y_cvx = simulate_gradient_descent('convex', start_x=3.5, learning_rate=0.08, steps=40)
            
            anns_cvx = [{'x': 1.0, 'y': 0.0, 'text': '唯一最优解', 'ax': 0, 'ay': -40}]
            
            fig2 = create_animated_plot(
                title="平稳滑向全局最优",
                x_range=[-1, 4.5],
                y_func_type="convex",
                path_x=path_x_cvx,
                path_y=path_y_cvx,
                annotations=anns_cvx
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.success("**总结**：凸函数像一个碗，没有陷阱。只要往下走，终点一定是最低点。")

# 运行主函数
if __name__ == "__main__":
    render()