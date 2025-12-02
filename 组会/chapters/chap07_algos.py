import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from utils import styler, play_animation 

# ==========================================
# 页面渲染逻辑
# ==========================================

def render():

    st.markdown("""
    优化算法是优化理论的“引擎”。模型定义的只是一个目标，而**如何到达那个最低点**，全靠优化算法。
    
    本章我们将从最简单的下山问题开始，逐步引入约束和非平滑项，最终推导出通用的复合优化框架。
    """)
    
    # --- 1. 学习路线图 (更新版) ---
    with st.expander("🗺️ 学习路线图 (点击展开)", expanded=True):
        st.markdown("""
        ### 1. ⛰️ 无约束优化 (Unconstrained)
        *   **场景**：身处连绵起伏的山脉，四周没有任何围栏。
        *   **核心挑战**：**地形险恶**。
            *   如果地形像狭长的峡谷（病态曲率），梯度下降会在谷壁间剧烈震荡，难以收敛。
        
        ### 2. 🚧 约束优化 (Constrained)
        *   **场景**：寻找最低点，但被限制在一个**围墙（可行域）**内。
        *   **核心挑战**：**边界阻挡**。
            *   **理论 (指南针)**：**拉格朗日乘子法**。告诉我们最优解处，想去的方向被墙壁的力完美抵消（梯度共线）。
            *   **算法 (步行者)**：**投影梯度法**。策略简单：先走一步，出界了再被墙“弹”回来。
        
        ### 3. 🧩 复合优化 (Composite)
        *   **场景**：目标函数由两部分拼接而成：$\text{光滑损失} + \text{有棱角的正则项}$。
        *   **核心挑战**：**尖点不可导**。
            *   地形中存在尖锐的棱角（如 L1 范数的尖峰），梯度失效。**近端梯度法 (Proximal)** 是处理它的通用框架。
        """)

    tab_frame, tab_unc, tab_con, tab_prox = st.tabs([
        "🧭 1. 迭代法通用框架", 
        "⚡ 2. 无约束 (GD vs Newton)", 
        "🧱 3. 约束优化 (Lagrange & Proj)", 
        "🧲 4. 复合优化 (Proximal GD)"
    ])
    
    # =================================================================
    # Tab 1: 通用框架 (交互式原理演示)
    # =================================================================
    with tab_frame:
        st.markdown("### 1. 迭代算法的本质：指南针与步伐")
        st.markdown(r"""
        几乎所有一阶优化算法都可以归纳为同一个公式：
        
        $$ \underbrace{x_{k+1}}_{\text{下一步}} = \underbrace{x_k}_{\text{当前位置}} + \underbrace{\alpha_k}_{\text{步长}} \cdot \underbrace{d_k}_{\text{方向}} $$
        
        * **方向 $d_k$**：通常由负梯度 $-\nabla f(x)$ 决定（最速下降方向）。
        * **步长 $\alpha_k$**：决定了你敢往那个方向走多远（Learning Rate）。
        """)
        
        col_demo, col_expl = st.columns([1.5, 1])
        with col_demo:
            # 交互：旋转搜索方向
            st.write("###### 🎮 交互：旋转你的搜索方向")
            angle = st.slider("偏离梯度方向的角度 (°)", -110, 110, 0, step=5)
            
            fig, ax = styler.get_fig_ax("搜索方向有效性验证")
            x = np.linspace(-2, 2, 100)
            y = x**2
            curr_x = 1.0
            grad = 2 * curr_x # f'(1) = 2 at x=1
            
            # 1. 画函数
            ax.plot(x, y, 'k-', alpha=0.4, lw=2, label='目标函数 $f(x)$')
            ax.scatter(curr_x, curr_x**2, color='black', s=80, zorder=5, label='当前点 $x_k$')
            
            # 2. 画标准负梯度 (Green)
            v_base = np.array([-1, -grad]) 
            v_base = v_base / np.linalg.norm(v_base)
            
            ax.arrow(curr_x, curr_x**2, v_base[0], v_base[1], head_width=0.1, color='#2ca02c', alpha=0.5, lw=3, label='负梯度 (最速下降)')
            
            # 3. 画用户方向 (Red/Blue)
            rad = np.radians(angle)
            # 旋转矩阵
            R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            v_user = R @ v_base
            
            # 判断是否上升 (夹角 > 90度)
            is_descent = abs(angle) < 90
            color = '#1f77b4' if is_descent else '#d62728'
            status = "有效下降" if is_descent else "错误！函数值上升"
            
            ax.arrow(curr_x, curr_x**2, v_user[0], v_user[1], head_width=0.1, color=color, lw=2, label=f'你的方向 ({status})')
            
            ax.legend(loc='upper center')
            ax.set_ylim(-0.5, 4.5)
            st.pyplot(fig)
            
        with col_expl:
            if is_descent:
                st.success(f"""
                **✅ 方向正确**
                
                夹角 {abs(angle)}° < 90°。
                虽然不如梯度方向下降得快，但只要是锐角，函数值依然在减少，算法能正常工作。
                """)
            else:
                st.error(f"""
                **❌ 方向错误**
                
                夹角 {abs(angle)}° > 90°。
                此时你在往山上走！优化算法会发散（Loss 变大）。
                """)

    # =================================================================
    # Tab 2: 无约束优化 (GD vs Newton 动态竞赛)
    # =================================================================
    with tab_unc:
        st.markdown("### 2. 地形决定命运：为什么 GD 会震荡？")
        st.markdown(r"""
        在深度学习中，我们经常听到“病态曲率 (Ill-conditioned curvature)”这个词。这指的是地形在一个方向非常陡峭，而在另一个方向非常平缓（像一个狭长的峡谷）。
        
        * **梯度下降 (First-Order)**：它是“近视眼”，只看脚下的坡度。在峡谷中，它会在陡峭的墙壁间来回**震荡**，很难沿着平缓的谷底前进。
        * **牛顿法 (Second-Order)**：它利用海森矩阵 (Hessian) 感知地形的**曲率**。它能“看穿”峡谷的形状，直接校正方向，沿着谷底直奔终点。
        """)
        
        col_ctrl, col_anim = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("#### ⚙️ 参数控制")
            kappa = st.slider("地形拉伸程度 (Condition Number)", 2.0, 15.0, 10.0, step=1.0, help="值越大，峡谷越细长，GD越难走")
            lr = st.number_input("学习率 (Learning Rate)", 0.01, 0.5, 0.15, step=0.01)
            
            st.warning(f"""
            **当前地形**: 
            $f(x) = 0.5(x_1^2 + {kappa:.0f}x_2^2)$
            
            $x_2$ 方向比 $x_1$ 陡峭 {kappa:.0f} 倍。
            """)
            
            run_race = st.button("▶️ 开始竞赛 (GD vs Newton)", key="btn_race")

        with col_anim:
            # 准备数据
            def func(x): return 0.5 * (x[0]**2 + kappa * x[1]**2)
            def grad(x): return np.array([x[0], kappa * x[1]])
            
            # 计算完整路径
            start_pos = np.array([4.0, 2.5])
            
            # Path GD
            path_gd = [start_pos]
            for _ in range(15):
                g = grad(path_gd[-1])
                path_gd.append(path_gd[-1] - lr * g)
            path_gd = np.array(path_gd)
            
            # Path Newton (1步收敛)
            H_inv = np.array([[1, 0], [0, 1/kappa]])
            p_newton = start_pos - H_inv @ grad(start_pos)
            path_nt = np.array([start_pos, p_newton] + [p_newton]*14) # 补齐长度方便动画
            
            def draw_race_frame(frame_idx):
                idx = int(frame_idx)
                fig, ax = styler.get_fig_ax(f"迭代步数: {idx}")
                
                # 画等高线背景
                X, Y = np.meshgrid(np.linspace(-5, 5, 80), np.linspace(-4, 4, 80))
                Z = 0.5 * (X**2 + kappa * Y**2)
                ax.contour(X, Y, Z, levels=[0.5, 2, 5, 10, 20, 30], colors='gray', alpha=0.3)
                
                # 画终点
                ax.scatter(0, 0, marker='*', s=200, color='gold', zorder=10, edgecolors='black')
                
                # 画 GD 轨迹 (Current History)
                h_gd = path_gd[:idx+1]
                ax.plot(h_gd[:,0], h_gd[:,1], 'r.-', alpha=0.6, label='Gradient Descent')
                ax.scatter(h_gd[-1,0], h_gd[-1,1], color='red', s=60) # 头
                
                # 画 Newton 轨迹
                h_nt = path_nt[:idx+1]
                ax.plot(h_nt[:,0], h_nt[:,1], 'b.--', alpha=0.6, label='Newton Method')
                ax.scatter(h_nt[-1,0], h_nt[-1,1], color='blue', s=60) # 头
                
                ax.set_xlim(-1, 5)
                ax.set_ylim(-3.5, 3.5)
                ax.legend(loc='upper right')
                return fig

            if run_race:
                # 播放动画
                play_animation(draw_race_frame, np.arange(15), interval=0.15)
            else:
                st.pyplot(draw_race_frame(0))
            
            if kappa > 12:
                st.caption("👀 注意观察：当地形非常狭长时，GD 即使还没有到达终点，步长也必须设得很小，否则就会发散；而牛顿法不受影响。")

    # =================================================================
    # Tab 3: 约束优化 (Lagrange + Projected GD) - 更新版
    # =================================================================
    with tab_con:
        st.markdown("### 3. 约束优化：理论与实践")
        
        # 使用 Radio 组件切换视角
        view_mode = st.radio(
            "选择视角：",
            ["📐 理论视角：拉格朗日乘子法 (KKT)", "🏃 算法视角：投影梯度法 (Projected GD)"],
            horizontal=True
        )

        # --- 视角 A: 拉格朗日乘子法 (几何直觉) ---
        if "理论" in view_mode:
            st.markdown(r"""
            #### 力的平衡：为什么梯度必须共线？
            
            当我们在约束边界上找到最优解时，会发生一件神奇的事情：
            **“我想去更低处的方向” ( $-\nabla f$ ) 正好被 “墙壁的法线方向” ( $\nabla g$ ) 顶住了。**
            
            这对应的数学公式就是 **KKT 条件** 的核心：
            $$ \nabla f(x^*) + \lambda \nabla g(x^*) = 0 $$
            
            *   $\nabla f$：目标函数的梯度（指向函数值增加的方向）。
            *   $\nabla g$：约束函数的梯度（垂直于约束表面）。
            *   $\lambda$：拉格朗日乘子（调节两个力的大小，使它们抵消）。
            """)

            col_lag_text, col_lag_plot = st.columns([1, 1.5])
            
            with col_lag_text:
                st.info("""
                **图示解读**：
                
                *   **蓝色圆圈**：目标函数 $f(x)$ 等高线（圆心是无约束极小值）。
                *   **红色直线**：约束条件 $g(x)=0$（我们必须站在这条线上）。
                *   **⭐ 星号**：约束下的最优解。
                
                **观察箭头**：
                在最优解处，蓝色的箭头（目标梯度）和红色的箭头（约束梯度）是**平行但反向**的。
                
                这意味着：*再往哪里走都不行了，除非穿墙。*
                """)

            with col_lag_plot:
                fig_lag, ax_lag = styler.get_fig_ax("拉格朗日乘子法的几何意义")
                
                # 1. 绘制等高线 f(x,y) = x^2 + y^2
                x = np.linspace(-1, 5, 100)
                y = np.linspace(-1, 5, 100)
                X, Y = np.meshgrid(x, y)
                Z = X**2 + Y**2
                ax_lag.contour(X, Y, Z, levels=[2, 8, 18, 32, 50], colors='cornflowerblue', alpha=0.5)
                
                # 2. 绘制约束 g(x,y) = x + y - 4 = 0 -> y = 4 - x
                y_line = 4 - x
                ax_lag.plot(x, y_line, 'r-', linewidth=3, label='约束墙壁 g(x)=0')
                
                # 3. 最优解 (2, 2)
                opt_x, opt_y = 2, 2
                ax_lag.scatter([opt_x], [opt_y], c='gold', s=150, zorder=10, edgecolors='k', label='约束最优解')
                
                # 4. 绘制梯度向量
                # 目标梯度 grad_f = (2x, 2y) -> (4, 4) at (2,2)
                # 归一化为了画图好看
                ax_lag.arrow(opt_x, opt_y, 0.8, 0.8, head_width=0.2, fc='blue', ec='blue', width=0.05)
                ax_lag.text(opt_x+0.9, opt_y+0.9, r'$\nabla f$ (想离开墙)', color='blue', fontsize=12, fontweight='bold')
                
                # 约束梯度 grad_g = (1, 1)
                # 反向画，模拟受力平衡
                ax_lag.arrow(opt_x, opt_y, -0.8, -0.8, head_width=0.2, fc='red', ec='red', width=0.05)
                ax_lag.text(opt_x-1.5, opt_y-1.0, r'$-\lambda \nabla g$ (墙的推力)', color='red', fontsize=12, fontweight='bold')
                
                ax_lag.set_xlim(0, 4.5)
                ax_lag.set_ylim(0, 4.5)
                ax_lag.legend(loc='lower left')
                ax_lag.set_aspect('equal')
                st.pyplot(fig_lag)

        # --- 视角 B: 投影梯度法 (算法过程) ---
        elif "算法" in view_mode:
            st.markdown(r"""
            #### 简单粗暴的策略：先走再说，出界再弹回
            
            虽然拉格朗日法很美，但在高维空间解方程组很难。实际计算中，我们常用**投影梯度法**：
            
            $$ x_{k+1} = \Pi_C (x_k - \alpha \nabla f(x_k)) $$
            
            1.  **Gradient Step**: 无视约束，先走一步。
            2.  **Projection Step**: 如果这一步跨出了墙外，就找墙上离你最近的点站上去。
            """)
            
            col_c1, col_c2 = st.columns([1, 1.5])
            with col_c1:
                st.success("""
                **动画演示**：
                
                *   **黑色虚线**：梯度尝试步（想直接去靶心，但出界了）。
                *   **红色箭头**：投影操作（被墙壁强制修正）。
                *   **蓝色方框**：可行域。
                """)
                run_proj = st.button("▶️ 播放投影过程", key="btn_proj")
                
            with col_c2:
                # 模拟数据
                path_proj = []
                curr = np.array([-0.5, -0.5])
                target = np.array([1.5, 1.5]) # 目标在盒子外
                
                # 生成模拟路径数据结构
                for i in range(8):
                    g = curr - target
                    temp = curr - 0.6 * g # 大步长制造出界
                    nxt = np.clip(temp, -1, 1) # 投影到 Box
                    path_proj.append({
                        'start': curr,
                        'temp': temp,
                        'final': nxt
                    })
                    curr = nxt
                    
                def draw_proj_frame(step_i):
                    idx = int(step_i)
                    fig, ax = styler.get_fig_ax(f"投影梯度法: Step {idx+1}")
                    
                    # 1. 绘制约束盒
                    rect = Rectangle((-1, -1), 2, 2, facecolor='#e3f2fd', edgecolor='#2196f3', lw=2, alpha=0.5, label='可行域 (约束)')
                    ax.add_patch(rect)
                    
                    # 2. 绘制等高线
                    X, Y = np.meshgrid(np.linspace(-1.5, 2.5, 60), np.linspace(-1.5, 2.5, 60))
                    Z = (X-1.5)**2 + (Y-1.5)**2
                    ax.contour(X, Y, Z, levels=10, colors='gray', alpha=0.2)
                    ax.scatter(1.5, 1.5, marker='x', color='gray', label='无约束极小值 (墙外)')
                    ax.scatter(1.0, 1.0, marker='*', s=150, color='gold', edgecolors='k', zorder=10, label='约束最优解')

                    # 3. 绘制历史
                    for k in range(idx):
                        p = path_proj[k]
                        ax.plot([p['start'][0], p['final'][0]], [p['start'][1], p['final'][1]], 'r.-', alpha=0.3)

                    # 4. 绘制当前步
                    current_move = path_proj[idx]
                    start = current_move['start']
                    temp = current_move['temp']
                    final = current_move['final']
                    
                    # 梯度步
                    ax.plot([start[0], temp[0]], [start[1], temp[1]], 'k--', label='1. 尝试步')
                    ax.scatter(temp[0], temp[1], c='black', alpha=0.5)
                    
                    # 投影步
                    if np.linalg.norm(temp - final) > 1e-4:
                        arrow = FancyArrowPatch(posA=tuple(temp), posB=tuple(final), arrowstyle='-|>', mutation_scale=15, color='#d32f2f', lw=2, label='2. 投影修正')
                        ax.add_patch(arrow)
                    
                    ax.scatter(final[0], final[1], c='#d32f2f', s=60, zorder=5)

                    ax.set_xlim(-1.2, 2.2)
                    ax.set_ylim(-1.2, 2.2)
                    ax.legend(loc='lower right', fontsize=8)
                    return fig

                if run_proj:
                    play_animation(draw_proj_frame, np.arange(len(path_proj)), interval=0.6)
                else:
                    st.pyplot(draw_proj_frame(0))

    # =================================================================
    # Tab 4: 复合优化 (Proximal 算子可视化)
    # =================================================================
    with tab_prox:
        st.markdown("### 4. 复合优化：处理“不可导”的通用框架")
        
        st.markdown(r"""
        很多现代优化问题（如 Lasso、稀疏编码）的目标函数由两部分组成：
                    
        $$ \min_x F(x) = \underbrace{f(x)}_{\text{平滑损失 (如 MSE)}} + \underbrace{g(x)}_{\text{非平滑正则 (如 L1)}} $$
        
        **痛点**：$g(x)$ (比如 L1 范数) 在 0 点是尖锐的，无法求导，导致传统的梯度下降失效。
        
        **解决方案 (Proximal Gradient)**：我们将问题分解为两步：
        1.  **梯度步 (Forward)**：先不管 $g(x)$，只沿着 $f(x)$ 的梯度走一步。
            $$ z_{k+1} = x_k - \alpha \nabla f(x_k) $$
        2.  **近端步 (Backward)**：寻找一个点，既靠近 $z_{k+1}$，又能让 $g(x)$ 尽量小。
            $$ x_{k+1} = \text{prox}_{\alpha g}(z_{k+1}) $$
        """)
        
        st.divider()
        st.subheader("🧪 交互实验：近端算子 (Proximal Operator) 的魔法")
        st.markdown("在这个实验中，我们的 $f(x)$ 总是指向原点。观察**不同的正则项 $g(x)$** 如何像“磁铁”或“墙壁”一样改变最终的落点。")

        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            prox_type = st.radio(
                "选择非平滑项 g(x)",
                ["None (无)", "L2 Norm (岭回归)", "L1 Norm (Lasso)", "Indicator Box (约束)"]
            )
            
            lam = st.slider("正则化强度 / 步长", 0.1, 2.0, 1.0)
            
            explanation = ""
            if prox_type == "L1 Norm (Lasso)":
                explanation = "L1 算子不仅会缩小数值，还会强制将较小的数值直接**截断为 0**。这就是为什么 Lasso 能产生稀疏解（特征选择）的原因。"
            elif prox_type == "Indicator Box (约束)":
                explanation = "当 g(x) 是指示函数时，近端算子退化为**投影**。这说明投影梯度法其实是近端梯度法的一个特例！"
            
            if explanation:
                st.info(explanation)
            
            st.markdown("""
            **图示说明**：
            1. 🟢 **绿线**：梯度步 (Gradient Step)。
            2. 🟠 **橙线**：近端步 (Proximal Step)，正则项产生的“拉力”。
            3. 🔴 **红点**：最终更新位置。
            """)

        with col_plot:
            # Setup
            start_pt = np.array([2.5, 2.0])
            grad_step = np.array([-1.0, -1.0]) # 假设梯度指向左下
            z_k = start_pt + grad_step # 中间点
            alpha = 1.0
            
            # 初始化 limit 变量，防止 UnboundLocalError
            limit = 0.0
            
            # Proximal Logic
            if prox_type == "None (无)":
                x_next = z_k
                geo_desc = "无修正：纯梯度下降"
                
            elif prox_type == "L2 Norm (岭回归)":
                # prox = z / (1 + lambda)
                x_next = z_k / (1 + alpha * lam)
                geo_desc = "权重衰减 (Weight Decay)：按比例收缩，方向不变"

            elif prox_type == "L1 Norm (Lasso)":
                # Soft Thresholding
                threshold = alpha * lam
                sign = np.sign(z_k)
                mag = np.maximum(np.abs(z_k) - threshold, 0)
                x_next = sign * mag
                geo_desc = "软阈值 (Soft Threshold)：数值减小，小值归零"

            elif prox_type == "Indicator Box (约束)":
                limit = max(0.2, 1.0 - alpha * lam * 0.5)
                x_next = np.clip(z_k, -limit, limit)
                geo_desc = "投影 (Projection)：超出范围被截断"

            # Visualization
            fig, ax = styler.get_fig_ax(f"Proximal Operator: {prox_type}")
            
            # Background contours
            ax.axhline(0, color='k', lw=0.5)
            ax.axvline(0, color='k', lw=0.5)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Constraint/Region Visualization
            if prox_type == "Indicator Box (约束)":
                 rect = Rectangle((-limit, -limit), 2*limit, 2*limit, 
                                  fill=True, color='#FFF3E0', alpha=0.5, label='约束区域')
                 ax.add_patch(rect)
            elif prox_type == "L1 Norm (Lasso)":
                 ax.text(0.1, 0.1, "L1 '磁力' 轴", color='orange', fontsize=8)

            # Points and Arrows
            ax.scatter(start_pt[0], start_pt[1], c='black', s=80, label='$x_k$ (起点)')
            
            # Step 1: Gradient
            ax.annotate("", xy=z_k, xytext=start_pt, arrowprops=dict(arrowstyle="->", color="green", lw=2))
            ax.text((start_pt[0]+z_k[0])/2, (start_pt[1]+z_k[1])/2 + 0.15, "1. 梯度步", color='green', fontweight='bold', ha='center')
            ax.scatter(z_k[0], z_k[1], c='green', alpha=0.4, s=50)

            # Step 2: Prox
            if not np.allclose(z_k, x_next):
                ax.annotate("", xy=x_next, xytext=z_k, arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=3))
                ax.text((z_k[0]+x_next[0])/2 + 0.1, (z_k[1]+x_next[1])/2, "2. 近端步", color='#FF6F00', fontweight='bold')
            
            ax.scatter(x_next[0], x_next[1], c='#D32F2F', s=120, zorder=10, label='$x_{k+1}$ (最终落点)')
            
            # Text Summary
            ax.text(0, -0.4, f"效果: {geo_desc}", fontsize=12, color='#333', backgroundcolor='#f0f0f0')
            
            # L1 特效：如果归零了，画个特效
            if prox_type.startswith("L1") and (x_next[0]==0 or x_next[1]==0):
                ax.text(x_next[0], x_next[1]-0.3, "稀疏化 (Zero)!", color='red', fontweight='bold', ha='center')

            ax.legend(loc='upper right')
            ax.set_xlim(-0.5, 3.0)
            ax.set_ylim(-0.5, 3.0)
            st.pyplot(fig)

if __name__ == "__main__":
    render()