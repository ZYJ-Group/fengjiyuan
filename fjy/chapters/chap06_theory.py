import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import styler 

def render():
    # --- 全局配置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 逻辑重构：增加第四个标签页“复合优化”
    tab_prob, tab_opt, tab_unc, tab_con, tab_comp = st.tabs(["🗺️ 问题的本质", "🎯 最优性条件", "⛰️ 无约束优化 (费马)", "🚧 有约束优化 (KKT)", "🧩 复合优化 (Proximal)"])

# --- Tab 1: 问题的本质 ---
    with tab_prob:
        st.markdown(r"""
        ### 1. 优化问题的数学表述 (Mathematical Formulation)
        
        最优化理论旨在从所有可能的方案中选择由目标函数定义的“最优”元素。一个标准的数学优化问题通常表述为如下的**一般形式**：
        
        $$
        \begin{aligned}
        \min_{x} \quad & f(x) \\
        \text{s.t.} \quad & x \in \Omega
        \end{aligned}
        $$
        
        该数学模型由以下核心要素构成：
        * **决策变量 (Decision Variable) $x$**：通常是一个 $n$ 维向量 $x \in \mathbb{R}^n$，代表我们需要优化的参数或系统状态。
        * **目标函数 (Objective Function) $f: \mathbb{R}^n \to \mathbb{R}$**：一个将决策变量映射到实数的标量函数。优化的目标是寻找使 $f(x)$ 最小化的 $x$。
        * **可行集 (Feasible Set) $\Omega$**：决策变量必须满足的约束条件的集合（如 $\Omega = \{x \mid c_i(x) \le 0, h_j(x) = 0\}$）。$\Omega$ 定义了搜索空间的几何边界。
        
        ---

        ### 2. 非凸性与极值分析 (Non-convexity & Extrema)
        若目标函数 $f(x)$ 或可行集 $\Omega$ 不满足凸性条件（Non-convex），地形将呈现复杂的多峰结构。此时，我们需要严格区分两种极值概念：
        
        * **局部极小值 (Local Minimum)**：存在一个邻域 $\mathcal{N}(x^*)$，使得对于所有 $x \in \mathcal{N}(x^*) \cap \Omega$，都有 $f(x^*) \le f(x)$。这对应于梯度下降法容易陷入的**吸引盆 (Basin of Attraction)**。
        * **全局极小值 (Global Minimum)**：对于可行域内所有 $x \in \Omega$，恒有 $f(x^*) \le f(x)$。这是优化算法试图逼近的理论下确界。
        """)
        
        col_text, col_plot = st.columns([1, 1.5])
        
        with col_text:
            st.info("""
            **说明**
            
            请观察右侧的目标函数 $f(x)$ 曲线：
            
            * **🟠 橙色点 (局部极小值)**：
                满足一阶最优性条件 $\nabla f(x) = 0$ 且海森矩阵正定 $\nabla^2 f(x) \succ 0$。但它仅在局部邻域内最优。基于梯度的算法极易在此处收敛并停滞。
                
            * **🔴 红色五角星 (全局极小值)**：
                函数的全局 $f_{\min}$ 所在位置。在非凸优化中，寻找该点的算法复杂度通常是 NP-Hard 的。
            """)
            
        with col_plot:
            fig1, ax1 = styler.get_fig_ax("非凸优化景观：局部最优 vs 全局最优")
            x = np.linspace(-2, 2.5, 400)
            y = x**4 - 2*x**2 + 0.5*x
            ax1.plot(x, y, 'b-', linewidth=2, label=r'目标函数 $f(x)$')
            
            local_min_x = -1.05  
            local_min_y = local_min_x**4 - 2*local_min_x**2 + 0.5*local_min_x
            ax1.scatter([local_min_x], [local_min_y], color='orange', s=100, zorder=10, label='局部极小值 (Local Min)')
            
            global_min_x = 0.93 
            global_min_y = global_min_x**4 - 2*global_min_x**2 + 0.5*global_min_x
            ax1.scatter([global_min_x], [global_min_y], color='red', s=120, marker='*', zorder=10, label='全局极小值 (Global Min)')
            
            ax1.set_ylim(-2, 4)
            ax1.set_xlabel(r"决策变量 $x$")
            ax1.set_ylabel(r"函数值 $f(x)$")
            ax1.legend()
            st.pyplot(fig1)

    # --- Tab 2: 最优性条件 & 存在性 (新增内容在这里) ---
    with tab_opt:
        st.markdown(r"""
        ### 3. 最优性条件 (Generalized Fermat's Rule)
        
        对于**无约束**的凸优化问题，点 $x^*$ 是全局极小值点的**充要条件**是：
                    
        $$ 0 \in \partial f(x^*) $$

        **几何解释 (支撑超平面)**：
        回顾次梯度的定义：$g$ 是次梯度意味着 $f(y) \ge f(x) + g^T(y-x)$。
        
        当 $0 \in \partial f(x^*)$ 时，意味着我们可以画出一个**水平的支撑超平面**。
        代入公式即得：$f(y) \ge f(x^*) + 0$，即对于任意 $y$，函数值都比 $x^*$ 处要高（或相等）。这正是全局最小值的定义。
        """)

        st.markdown("---") # 分割线

        # --- 新增：最优解的存在条件 ---
        st.subheader("4. 最优解总是存在吗？ (Existence Conditions)")
        
        col_ex_text, col_ex_viz = st.columns([1.2, 1])
        
        with col_ex_text:
            st.markdown(r"""
            满足 $0 \in \partial f(x)$ 只是找到了候选点，但有些问题**根本没有解**。
            
            为了保证最优解 $x^*$ 一定存在，通常需要满足以下两个条件之一 (Weierstrass 定理推广)：
            
            #### ✅ 条件 A：定义域紧致 (Compactness)
            如果问题的定义域 $\mathcal{X}$ 是**有界且闭合**的（例如一个闭区间 $[a, b]$），且函数连续，则最小值一定存在。
            
            #### ✅ 条件 B：函数强制 (Coercivity)
            如果是无约束优化 ($x \in \mathbb{R}^n$)，函数必须是**强制的 (Coercive)**：
            $$ \lim_{\|x\| \to \infty} f(x) = +\infty $$
            
            > **通俗解释**：
            > * **强制函数**像一个“碗”，两头都往上翘，中间一定有个底。
            > * **非强制函数**像一个“滑梯”或“渐近线”，可能永远在下降，取不到终点。
            """)
            
            st.info(r"**反例**：$f(x) = e^{-x}$ 是凸函数，但 $\lim_{x\to \infty} f(x) = 0$，最小值 0 永远取不到（无解）。")

        with col_ex_viz:
            # --- 可视化：强制性 vs 非强制性 ---
            fig_exist, ax_ex = styler.get_fig_ax("可视化：强制性 (Coercivity)")
            
            x_ex = np.linspace(-1.5, 2.5, 200)
            
            # 1. 强制函数 (Coercive)
            y_coercive = (x_ex - 0.5)**2 + 0.5
            ax_ex.plot(x_ex, y_coercive, 'g-', linewidth=2.5, label=r'强制: $f(x)=(x-0.5)^2$')
            ax_ex.scatter([0.5], [0.5], c='green', s=80, zorder=5, label='存在最小值')
            
            # 2. 非强制函数 (Non-coercive)
            y_non_coercive = np.exp(-x_ex)
            ax_ex.plot(x_ex, y_non_coercive, 'r--', linewidth=2.5, label=r'非强制: $f(x)=e^{-x}$')
            
            # 装饰
            ax_ex.set_ylim(0, 4)
            ax_ex.set_xlim(-1.5, 2.5)
            ax_ex.set_xlabel("x")
            ax_ex.set_ylabel("f(x)")
            
            # 添加文字说明
            ax_ex.text(1.8, 3.5, "两端趋于 $+\infty$\n(有底)", color='green', fontsize=9, ha='center')
            ax_ex.text(2.0, 0.3, r"趋于 0 但取不到", color='red', fontsize=9, ha='center')
            
            ax_ex.legend(loc='upper left', fontsize=9)
            st.pyplot(fig_exist)

        st.markdown("---")

        # --- 新增：有约束的最优性条件 ---
        st.subheader("5. 有约束情况 (Constrained Optimality)")
        
        col_con_text, col_con_viz = st.columns([1.2, 1])
        
        with col_con_text:
            st.markdown(r"""
            如果问题加上了约束：$\min f(x) \quad \text{s.t.} \quad x \in \mathcal{C}$。
            
            此时，$x^*$ 是全局最小值的**充要条件**变为：
            
            $$ 0 \in \partial f(x^*) + N_\mathcal{C}(x^*) $$
            
            或者写作：
            $$ -\partial f(x^*) \cap N_\mathcal{C}(x^*) \neq \emptyset $$
            
            **核心概念：法锥 (Normal Cone) $N_\mathcal{C}(x)$**
            法锥包含了所有与集合 $\mathcal{C}$ 在点 $x$ 处“垂直向外”的向量。
            
            **直观翻译**：
            > **“任何能让你函数值下降的方向（负梯度），都会让你直接掉出可行域（进入法锥方向）。”**
            > 所以你被卡在边界上了，没法再走了。
            """)

        with col_con_viz:
            fig_nc, ax_nc = styler.get_fig_ax("几何直觉：梯度被法锥抵消")
            
            # 画一个简单的可行域 (圆形)
            theta = np.linspace(0, 2*np.pi, 100)
            ax_nc.fill(np.cos(theta), np.sin(theta), color='lightgreen', alpha=0.3, label='可行域 C')
            ax_nc.plot(np.cos(theta), np.sin(theta), color='green', linestyle='--')
            
            # 选一个边界点
            pt_x = np.cos(np.pi/4)
            pt_y = np.sin(np.pi/4)
            ax_nc.scatter([pt_x], [pt_y], color='black', s=80, zorder=10, label='最优解 x*')
            
            # 画法锥方向 (垂直向外)
            ax_nc.arrow(pt_x, pt_y, 0.5, 0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax_nc.text(pt_x+0.6, pt_y+0.6, r"法锥 $N_C(x^*)$", color='red', fontsize=10)
            
            # 画负梯度方向 (想去下降的方向)
            # 在最优解处，负梯度必须指向法锥内部（即指向可行域外）
            ax_nc.arrow(pt_x, pt_y, 0.5, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linestyle=':')
            ax_nc.text(pt_x+0.1, pt_y+0.8, r"$-\nabla f(x^*)$", color='blue', fontsize=10)
            ax_nc.text(pt_x+0.2, pt_y+0.65, "(下降方向)", color='blue', fontsize=8)

            ax_nc.set_xlim(-1.5, 2.5)
            ax_nc.set_ylim(-1.5, 2.5)
            ax_nc.set_aspect('equal')
            ax_nc.legend(loc='lower left', fontsize=8)
            st.pyplot(fig_nc)

        st.markdown("---")

        # --- 新增：复合优化的最优性条件 ---
        st.subheader("6. 复合优化情况 (Composite Optimality)")
        
        col_comp_text, col_comp_viz = st.columns([1.2, 1])
        
        with col_comp_text:
            st.markdown(r"""
            对于复合问题：$\min F(x) = f(x) + g(x)$，其中 $f$ 光滑，$g$ 非光滑（如 L1 范数）。
            
            全局最优解 $x^*$ 的**充要条件**是：
            
            $$ 0 \in \nabla f(x^*) + \partial g(x^*) $$
            
            或者写作：
            $$ -\nabla f(x^*) \in \partial g(x^*) $$
            
            **直观解释**：
            > **“光滑部分的推力（负梯度）被非光滑部分的阻力（次梯度）完全抵消。”**
            > 就像你在推一个箱子（梯度下降），但地面的摩擦力（正则项次梯度）太大，箱子推不动了，这就达到了平衡点。
            """)

        with col_comp_viz:
            fig_comp, ax_comp = styler.get_fig_ax("几何直觉：梯度被次梯度抵消")
            
            # 画一个简单的 1D 示意图
            x_vals = np.linspace(-1.5, 1.5, 100)
            # f(x) = (x-0.5)^2 -> grad = 2(x-0.5). At x=0, grad = -1.
            # g(x) = |x| -> subgrad at 0 is [-1, 1].
            # Optimal is at x=0 because -grad(0) = 1 \in [-1, 1]? No.
            # Let f(x) = 0.5(x-1)^2. grad(x) = x-1.
            # At x=0.5, grad = -0.5. subgrad(|x|) = sign(0.5) = 1. Sum != 0.
            # Let's visualize the condition at x=0.
            
            # 模拟 x=0 处的情况
            # f(x) 的负梯度指向右边
            ax_comp.arrow(0, 0, 0.8, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue', width=0.02)
            ax_comp.text(0.2, 0.1, r"$-\nabla f(0)$ (推力)", color='blue')
            
            # g(x) = |x| 的次梯度区间 [-1, 1]
            # 画出次梯度集合产生的“阻力范围”
            ax_comp.hlines(0, -1, 1, colors='orange', linewidth=5, alpha=0.5, label=r'$\partial g(0) = [-1, 1]$')
            ax_comp.text(-0.8, -0.2, "次梯度集合 (阻力范围)", color='orange')
            
            # 结论
            ax_comp.scatter([0], [0], c='black', s=100, zorder=10)
            ax_comp.text(0, 0.3, "最优解 x*=0\n(推力落在阻力范围内)", ha='center', fontweight='bold')
            
            ax_comp.set_xlim(-1.5, 1.5)
            ax_comp.set_ylim(-0.5, 0.5)
            ax_comp.set_yticks([])
            ax_comp.legend(loc='lower right', fontsize=8)
            st.pyplot(fig_comp)
    # --- Tab 3: 无约束优化 ---
    with tab_unc:
        st.markdown(r"""
        ### 5. 这里的规则很简单
        如果没有围栏限制（无约束），我们怎么知道自己到了谷底？
        
        想象你在山谷中滑行：
        1.  **一阶条件 (平地)**：如果你到了谷底，脚下的地一定是平的（切线水平）。
            $$ \nabla f(x) = 0 $$
        2.  **二阶条件 (形状)**：光是地平还不够（山顶也是平的）。地必须是**向上弯曲**的（像个碗）。
            $$ \nabla^2 f(x) \succ 0 $$
        """)
        
        col_deriv_text, col_deriv_plot = st.columns([1, 1.5])
        
        with col_deriv_text:
            st.warning(r"""
            **关键点辨析**：

            * **驻点 (Stationary Point)**：
            满足 $\nabla f = 0$ 的点。它只是一个候选项，可能是极小值、极大值，也可能是“鞍点”（不凹也不凸的地方）。

            * **如何区分？** 看二阶导数（曲率）。
            * $f''(x) > 0$：开口向上，像个碗底 😄 -> **极小值**
            * $f''(x) < 0$：开口向下，像个山顶 ☹️ -> **极大值**
            """)
        
        with col_deriv_plot:
            fig2, (ax_f, ax_df) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            for ax in [ax_f, ax_df]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, linestyle=':', alpha=0.6)

            x = np.linspace(-1.5, 2.5, 200)
            y = (1/3)*x**3 - 0.5*x**2 - 2*x
            dy = x**2 - x - 2
            
            ax_f.plot(x, y, 'k-', linewidth=2, label=r'$f(x)$')
            ax_f.set_title("函数图像：寻找谷底", fontsize=12)
            ax_f.scatter([2], [(1/3)*8 - 2 - 4], c='green', s=100, zorder=5, label='极小值')
            ax_f.text(2, -4, "谷底\n(平坦且凹陷)", ha='center', va='top', color='green', fontweight='bold')
            
            ax_df.plot(x, dy, 'b--', linewidth=1.5, label=r"一阶导 $\nabla f$ (斜率)")
            ax_df.axhline(0, color='gray', linewidth=1)
            ax_df.scatter([2], [0], c='green', s=60)
            ax_df.annotate(r"斜率为 0", xy=(2, 0), xytext=(1.0, 2), arrowprops=dict(facecolor='black', arrowstyle='->'))
            
            ax_df.set_xlabel("x")
            ax_df.legend(loc='upper left')
            st.pyplot(fig2)

    # --- Tab 3: 有约束优化 ---
    with tab_con:
        st.markdown(r"""
        ### 6. 如果有围栏怎么办？
        现在的目标变成了：$\min f(x)$ s.t. $c(x) = 0$。
        
        想象你在山坡上想去谷底，但是被一堵墙（约束 $c(x)=0$）挡住了，你必须沿着墙走。
        
        ---
        
        ### 7. 什么时候停止？(直观理解)
        当你沿着墙走，走到**不能再让函数值下降**的位置时，你就在最优解了。
        
        > **几何直觉**：此时，你环顾四周，发现目标函数的“下降方向”（负梯度）正对着墙壁。
        > * 如果你还要往低处走，就必须穿墙而过（不被允许）。
        > * 如果你沿着墙走，高度不会降低（甚至会升高）。
        > * 所以，你卡住了。这就是最优解。
        
        ---

        ### 8. 数学表达：拉格朗日乘子法
        上述的几何直觉，翻译成数学语言就是：**目标函数的梯度与约束函数的梯度共线**。
        
        $$ \nabla f(x^*) + \lambda \nabla c(x^*) = 0 $$
        
        或者写作：
        $$ \nabla f(x^*) = -\lambda \nabla c(x^*) $$

        **公式深度拆解：**
        * **$\nabla f$ (目标梯度)**：垂直于目标函数的等高线，指向函数值增长最快的方向。
        * **$\nabla c$ (约束梯度)**：垂直于约束曲面（墙壁），是墙的法线方向。
        * **$\lambda$ (拉格朗日乘子)**：这只是一个**缩放系数**。它调节两个向量的长度，使它们大小相等、方向相反（平衡）。
        
        **结论**：
        这解释了为什么在最优解处，**目标函数的等高线必须与约束曲线相切**。如果不相切（而是相交），说明沿着约束曲线还有下降的空间，那就不是最优解。
        """)
        
        col_kkt_text, col_kkt_plot = st.columns([1, 1.2])
        
        with col_kkt_text:
            st.info("""
            **图示解读**：
            
            * **蓝色圆圈**：等高线。圆心在 (0,0) 是真正的谷底。
            * **红色直线**：约束墙壁。我们必须站在红线上。
            * **黑色点**：红线上离圆心最近的点（最优解）。
            
            **看箭头！**
            在最优解处，蓝色的箭头（目标梯度）和红色的箭头（约束梯度）是**共线**的。
            这意味着：*“我想去的方向，正好被墙顶回来了。”*
            """)
            
        with col_kkt_plot:
            fig3, ax3 = styler.get_fig_ax("约束优化的几何意义：梯度共线")
            x_grid = np.linspace(-1, 5, 100)
            y_grid = np.linspace(-1, 5, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = X**2 + Y**2
            
            x_line = np.linspace(0, 4.5, 100)
            y_line = 4 - x_line
            
            contours = ax3.contour(X, Y, Z, levels=[2, 8, 18, 32, 50], colors='cornflowerblue', alpha=0.6)
            ax3.clabel(contours, inline=True, fmt='f=%.0f')
            ax3.plot(x_line, y_line, 'r-', linewidth=3, label='约束墙壁 (可行域)')
            
            opt_x, opt_y = 2, 2
            ax3.scatter([opt_x], [opt_y], c='black', s=100, zorder=10, label='约束最优解')
            
            ax3.arrow(opt_x, opt_y, 0.5, 0.5, head_width=0.15, head_length=0.15, fc='blue', ec='blue', width=0.03)
            ax3.text(opt_x+0.6, opt_y+0.6, r'$\nabla f$ (想去反方向)', color='blue', fontsize=10)
            
            ax3.arrow(opt_x, opt_y, 0.5, 0.5, head_width=0.15, head_length=0.15, fc='red', ec='red', width=0.03, alpha=0.5)
            ax3.text(opt_x+0.1, opt_y-0.4, r'$\nabla c$ (墙的法线)', color='red', fontsize=10)
            
            ax3.set_xlim(0, 4.5)
            ax3.set_ylim(0, 4.5)
            ax3.set_aspect('equal')
            ax3.legend(loc='lower left')
            st.pyplot(fig3)

# --- Tab 4: 复合优化 (Composite Optimization) ---
    with tab_comp:
        st.markdown(r"""
        ### 9. 复合优化：当光滑遇见棱角
        
        在实际应用（如信号处理、图像去噪、低秩矩阵恢复）中，我们经常遇到这种形式的目标函数：
        
        $$ \min_x \quad F(x) = \underbrace{f(x)}_{\text{光滑部分}} + \underbrace{g(x)}_{\text{非光滑部分}} $$
        
        *   **$f(x)$**：通常是数据拟合项（Loss），它是光滑的，我们可以轻松计算梯度 $\nabla f(x)$。
        *   **$g(x)$**：通常是正则化项或指示函数。它可能在某些点**不可导**（有尖角），比如 $|x|$ 在 $x=0$ 处。

        ---
        ### 10. 核心矛盾：尖角处没有梯度
        
        如果我们直接对整体 $F(x)$ 求导，会发现在尖角处导数未定义。
        *   **左图展示了困境**：由 $f(x)$ 和 $g(x)$ 叠加后的函数 $F(x)$ 底部可能是一个尖锐的 V 字形。
        *   在尖点处，计算机无法计算“梯度”，传统的梯度下降法会在这里卡住或震荡。

        ### 11. 解决方案：近端梯度下降 (Proximal Gradient)

        数学家设计了一种**两步走**的策略（Forward-Backward Splitting）：
        
        1.  **Forward (显式梯度步)**：先忽略非光滑的 $g(x)$，只沿着 $f(x)$ 的梯度走一步。
            $$ z_k = x_k - \eta \nabla f(x_k) $$
        2.  **Backward (隐式近端步)**：在 $z_k$ 的基础上，寻找一个点，既离 $z_k$ 不太远，又能让 $g(x)$ 尽可能小。这步操作叫 **Proximal Operator**。
            $$ x_{k+1} = \text{prox}_{\eta g}(z_k) $$
        """)
        
        col_prob, col_mech = st.columns([1, 1])
        
        with col_prob:
            st.markdown("#### 📉 视觉直觉：函数的叠加")
            st.info("观察由光滑的抛物线（蓝色）和有尖角的绝对值（橙色）叠加而成的总目标函数（黑色）。")
            
            fig_1d, ax_1d = plt.subplots(figsize=(6, 4))
            
            x = np.linspace(-2.5, 2.5, 500)
            f_x = 0.5 * (x - 1)**2  # 光滑部分：抛物线，中心在 1
            g_x = 0.8 * np.abs(x)   # 非光滑部分：绝对值 V 形
            F_x = f_x + g_x         # 复合函数
            
            ax_1d.plot(x, f_x, 'b--', alpha=0.4, label=r'$f(x)$: 光滑抛物线')
            ax_1d.plot(x, g_x, 'orange', alpha=0.4, linestyle='--', label=r'$g(x)$: 非光滑尖角')
            ax_1d.plot(x, F_x, 'k-', linewidth=2.5, label=r'$F(x)$ (总目标)')
            
            # 标注尖点
            ax_1d.scatter([0], [0.5], color='red', s=50, zorder=5)
            ax_1d.annotate("尖点 (Kink)\n不可导!", xy=(0, 0.5), xytext=(-1.5, 2.0),
                           arrowprops=dict(facecolor='black', arrowstyle='->'),
                           fontsize=10, fontweight='bold', color='red')
            
            # 标注最优点
            min_idx = np.argmin(F_x)
            min_x = x[min_idx]
            min_y = F_x[min_idx]
            ax_1d.scatter([min_x], [min_y], color='green', s=80, marker='*', zorder=10)
            ax_1d.text(min_x, min_y - 0.5, "全局最优", ha='center', color='green', fontweight='bold')

            ax_1d.set_title("1D 视角：$f(x) + g(x)$ 的形状")
            ax_1d.legend()
            ax_1d.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_1d)

        with col_mech:
            st.markdown("#### ⚙️ 算法机制：Prox 算子")
            st.warning(r"""
            **Prox 算子到底做了什么？**
            
            它的定义是解决一个小型的优化问题：
            $$ \text{prox}_g(v) = \arg\min_x \left( g(x) + \frac{1}{2}\|x - v\|^2 \right) $$
            
            **直白翻译**：
            Prox 算子试图找到一个平衡点 $x$：
            1.  它想让 $g(x)$ 小（满足正则化要求）。
            2.  它又不想离原来的位置 $v$ 太远（保持梯度下降的成果）。
            """)
            
            # --- 绘图：Proximal Operator 的物理意义 ---
            fig_prox, ax_prox = plt.subplots(figsize=(6, 4))
            
            # 模拟一个 v 点（梯度步的结果）
            v_point = 1.5
            x_vals = np.linspace(-0.5, 2.5, 100)
            
            # 绘制两项的权衡
            # 1. 距离项 (x-v)^2
            dist_term = 0.5 * (x_vals - v_point)**2
            # 2. 正则项 g(x) (假设是 L1)
            reg_term = 0.8 * np.abs(x_vals)
            # 3. 总和
            prox_obj = dist_term + reg_term
            
            ax_prox.plot(x_vals, dist_term, 'b:', label=r'距离代价 $\frac{1}{2}\|x-v\|^2$')
            ax_prox.plot(x_vals, reg_term, 'orange', linestyle=':', label=r'正则代价 $g(x)$')
            ax_prox.plot(x_vals, prox_obj, 'r-', linewidth=2, label='Prox 目标函数')
            
            # 标记 v 和 prox(v)
            prox_solution = x_vals[np.argmin(prox_obj)]
            
            ax_prox.axvline(v_point, color='blue', alpha=0.3, linestyle='--')
            ax_prox.text(v_point, 3.5, "输入 v\n(梯度步结果)", color='blue', ha='center', fontsize=9)
            
            ax_prox.axvline(prox_solution, color='red', alpha=0.3, linestyle='--')
            ax_prox.scatter([prox_solution], [np.min(prox_obj)], color='red', s=100, zorder=10)
            ax_prox.annotate("输出 prox(v)\n(折中点)", xy=(prox_solution, np.min(prox_obj)), xytext=(prox_solution-1.0, 1.5),
                             arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')

            ax_prox.set_title(f"Prox 如何修正：从 v={v_point} 拉回到 {prox_solution:.2f}")
            ax_prox.set_ylim(0, 4)
            ax_prox.legend(loc='upper left', fontsize=8)
            ax_prox.grid(True, linestyle=':', alpha=0.5)
            
            st.pyplot(fig_prox)