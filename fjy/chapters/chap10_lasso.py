import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import platform
import time  # 需要导入 time 模块

# ==========================================
# 0. 基础配置 (解决中文乱码)
# ==========================================
def configure_fonts():
    """配置 Matplotlib 中文字体"""
    system_name = platform.system()
    fonts = []
    if system_name == "Windows":
        fonts = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    elif system_name == "Darwin":  # Mac
        fonts = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
    else:  # Linux / Streamlit Cloud
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
    
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心渲染函数
# ==========================================
def render():
    st.markdown("## 🎯 LASSO 回归：做减法的艺术")
    
    # --- 1. 核心理论部分 (保留内容) ---
    st.markdown(r"""
    **LASSO** 全称是 *Least Absolute Shrinkage and Selection Operator*。
    
    在线性回归的基础上，给目标函数加了一个**惩罚项**：
    
    $$
    \min_{w} \underbrace{\frac{1}{2n} \|y - Xw\|_2^2}_{\text{1. 拟合误差 (MSE)}} + \underbrace{\alpha \|w\|_1}_{\text{2. 惩罚项 (L1 范数)}}
    $$

    *   **第一部分**：负责让模型预测得准。
    *   **第二部分**：负责让模型参数（系数 $w$）的**绝对值之和**尽可能小。
    
    👉 **它的神奇之处**：当 $\alpha$（惩罚力度）足够大时，LASSO 会强制把那些**不重要的特征系数直接压缩为 0**。这意味着它能自动帮我们筛选出有用的数据特征。
    """)

    st.divider()

    # --- 2. 几何直观演示 (上帝视角) ---
    st.subheader("📐 深度解析：为什么 LASSO 能把系数变成 0？")
    
    st.markdown(r"""
    为了理解稀疏性，我们换一个**“上帝视角”**来看系数空间。
    我们将优化问题转化为：**在约束范围 $\|w\|_1 \le C$ 内，寻找 MSE 最小的点**。
    
    *   **背景等高线**：代表误差 (MSE)。越靠近红色十字，误差越小。
    *   **绿色菱形**：代表 LASSO 的约束范围。
    
    👇 **点击播放按钮，观察随着约束半径 C 减小（惩罚变大），解是如何移动到坐标轴上的：**
    """)

    col_ctrl, col_plot = st.columns([1, 2])

    # 定义网格 (用于画背景等高线) - 移到循环外只计算一次
    w1_range = np.linspace(-1, 5, 100)
    w2_range = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # 【修改 1】定义损失函数 (MSE) - 确保中心在 (3, 2)
    # 这是一个倾斜的椭圆山谷
    MSE = 1.5*(W1 - 3)**2 + (W2 - 2)**2 + 0.5*(W1-3)*(W2-2)

    # --- 控制区 ---
    with col_ctrl:
        st.write("#### 🎛️ 动画控制")
        run_anim = st.button("▶️ 播放收缩过程", type="primary", help="点击开始动态演示约束收缩过程")
        compare_ridge = st.checkbox("对比 Ridge (L2 圆形约束)", value=False, help="Ridge 的约束是圆形的，没有尖角")
        
        st.markdown("---")
        st.markdown("**📊 实时指标**")
        # 创建指标占位符
        metric_c = st.empty()
        metric_w1 = st.empty()
        metric_w2 = st.empty()
        
        st.info("""
        **观察重点**：
        注意当绿色菱形的**尖角**触碰到等高线时，
        系数 $w_2$ 会突然变成 0。
        """)

    # --- 绘图区占位符 ---
    with col_plot:
        plot_placeholder = st.empty()

    # --- 核心逻辑封装 ---
    def solve_and_draw(C_val):
        # 1. LASSO 求解 (菱形边界)
        # 使用参数方程遍历菱形边界 (只关注右半平面，因为最优解在右侧)
        t = np.linspace(0, 1, 200)
        b1_w1, b1_w2 = C_val*t, C_val*(1-t)       # 第一象限
        b4_w1, b4_w2 = C_val*t, C_val*(t-1)       # 第四象限
        boundary_w1 = np.concatenate([b1_w1, b4_w1])
        boundary_w2 = np.concatenate([b1_w2, b4_w2])
        
        # 【修改 2】计算 LASSO MSE - 公式必须与背景 MSE 完全一致 (中心 3, 2)
        boundary_mse = 1.5*(boundary_w1 - 3)**2 + (boundary_w2 - 2)**2 + 0.5*(boundary_w1-3)*(boundary_w2 - 2)
        
        min_idx = np.argmin(boundary_mse)
        lasso_w1, lasso_w2 = boundary_w1[min_idx], boundary_w2[min_idx]
        is_sparse = abs(lasso_w2) < 0.05

        # 2. Ridge 求解 (圆形边界) - 仅当勾选时计算
        ridge_w1, ridge_w2 = None, None
        if compare_ridge:
            theta = np.linspace(0, 2*np.pi, 360)
            c_w1 = C_val * np.cos(theta)
            c_w2 = C_val * np.sin(theta)
            # 【修改 3】计算 Ridge MSE - 同样修正中心为 (3, 2)
            c_mse = 1.5*(c_w1 - 3)**2 + (c_w2 - 2)**2 + 0.5*(c_w1-3)*(c_w2 - 2)
            min_c_idx = np.argmin(c_mse)
            ridge_w1, ridge_w2 = c_w1[min_c_idx], c_w2[min_c_idx]

        # 3. 更新左侧指标
        metric_c.metric("约束半径 C", f"{C_val:.2f}")
        metric_w1.metric("LASSO w1 (主特征)", f"{lasso_w1:.3f}")
        
        # 动态显示 w2 对比
        if compare_ridge:
            metric_w2.markdown(f"""
            **系数 w2 (噪声)**:
            * 🟢 **LASSO**: `{lasso_w2:.3f}` (已归零)
            * 🔶 **Ridge**: `{ridge_w2:.3f}` (仍保留)
            """)
        else:
            metric_w2.metric("LASSO w2 (噪声)", f"{lasso_w2:.3f}", 
                            delta="已归零 (稀疏)!" if is_sparse else "未归零", 
                            delta_color="inverse")

        # 4. 绘图逻辑
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制 MSE 等高线
        levels = [0.5, 2, 5, 10, 20, 30]
        contour = ax.contour(W1, W2, MSE, levels=levels, colors='gray', alpha=0.4, linestyles='--')
        ax.clabel(contour, inline=True, fontsize=8, fmt='MSE=%.1f')
        
        # 【修改 4】标记 OLS 最优解 - 位置修正为 (3, 2)
        ax.scatter([3], [2], c='red', marker='x', s=100, label='无正则最优解 (3, 2)', zorder=10)

        # --- 绘制 LASSO (绿色) ---
        diamond = Polygon([(-C_val, 0), (0, C_val), (C_val, 0), (0, -C_val)], 
                          facecolor='#2ca02c', alpha=0.2, edgecolor='#2ca02c', linewidth=2, label='LASSO 约束')
        ax.add_patch(diamond)
        # 绘制 LASSO 解点
        ax.scatter([lasso_w1], [lasso_w2], c='#2ca02c', s=200, zorder=20, edgecolors='white', linewidth=2, label='LASSO 解 (稀疏)')

        # --- 绘制 Ridge (橙色) ---
        if compare_ridge:
            circle = Circle((0, 0), C_val, facecolor='none', edgecolor='#ff7f0e', linestyle='-.', linewidth=2, label='Ridge 约束')
            ax.add_patch(circle)
            # 绘制 Ridge 解点
            ax.scatter([ridge_w1], [ridge_w2], c='#ff7f0e', marker='^', s=200, zorder=20, edgecolors='white', linewidth=2, label='Ridge 解 (非稀疏)')
            
            # 添加辅助线展示 Ridge 并没有归零
            if abs(ridge_w2) > 0.1:
                ax.vlines(ridge_w1, 0, ridge_w2, colors='#ff7f0e', linestyles=':', alpha=0.8)
                ax.text(ridge_w1+0.1, ridge_w2/2, f"w2={ridge_w2:.2f}", color='#ff7f0e', fontsize=10, fontweight='bold')

        # 动态注释
        if is_sparse:
            ax.annotate('LASSO 触底归零!', xy=(lasso_w1, lasso_w2), xytext=(lasso_w1-2.5, lasso_w2+1.5),
                        arrowprops=dict(facecolor='#2ca02c', shrink=0.05), fontsize=12, fontweight='bold', color='#2ca02c')
        
        ax.set_xlim(-1, 5)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel("系数 $w_1$", fontsize=11)
        ax.set_ylabel("系数 $w_2$", fontsize=11)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_title(f"几何博弈: C={C_val:.2f}", fontsize=14)
        
        return fig

    # --- 动画执行 ---
    if run_anim:
        # 生成从大到小的 C 值序列 (模拟收缩)
        c_values = np.linspace(4.0, 0.5, 40)
        for c in c_values:
            fig = solve_and_draw(c)
            plot_placeholder.pyplot(fig)
            plt.close(fig) # 释放内存
            time.sleep(0.05) # 控制帧率
    else:
        # 默认显示一个静态帧 (C=1.2)
        fig = solve_and_draw(1.2)
        plot_placeholder.pyplot(fig)
        plt.close(fig)

    # --- 3. 结论部分 (保留) ---
    if compare_ridge:
        st.warning("💡 **对比结论**：注意看圆形的 Ridge (橙色虚线)。圆形太圆滑了，它几乎永远只会碰到等高线的**侧边**，而很难刚好碰到坐标轴。这就是为什么 $L_2$ 正则化无法产生稀疏解。")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="LASSO 原理深度解析")
    configure_fonts()
    render()