import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import styler, get_isar_data, compute_rd_image, CS_ISAR_Solver, plot_isar_comparison

def render():
    # --- 全局配置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    # Tab 分层结构
    tab_theory, tab_sim, tab_analysis = st.tabs(["🧩 通用算法原理", "🚀 ISAR 应用仿真", "⚖️ 稳定性分析"])

    # ==========================================
    # Tab 1: 算法原理 (通用化重写)
    # ==========================================
    with tab_theory:
        st.markdown(r"""
        ### 1. 核心思想：分而治之 (Divide and Conquer)
        ADMM (交替方向乘子法) 是一种解决大规模、非平滑优化问题的强大框架。它的核心哲学不是“硬解”复杂问题，而是将一个大问题拆解为两个较小的子问题，交替求解。
        
        假设我们要解如下优化问题：
        $$ \min_x f(x) + g(x) $$
        
        *   **$f(x)$**：通常是平滑的损失函数（如最小二乘误差、对数似然）。
        *   **$g(x)$**：通常是非平滑的正则项或约束（如 L1 范数稀疏约束、全变分 TV、非负约束）。

        直接同时处理 $f$ 和 $g$ 很难。ADMM 使用 **“变量分裂” (Variable Splitting)** 技巧，引入辅助变量 $z$：
        
        $$ \min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad x - z = 0 $$
        
        现在，我们将难题拆分了：让 $x$ 专门处理函数 $f$，让 $z$ 专门处理函数 $g$，然后通过一个“协调者”迫使它们达成共识。
        
        ### 2. 宏观架构：三个步骤
        ADMM 的迭代过程可以被视为一个 **“协调机制”**：
        
        1.  **x-update (本地优化 A)**：
            $$ x^{k+1} := \text{argmin}_x \left( f(x) + \frac{\rho}{2} \|x - z^k + u^k\|_2^2 \right) $$
            > $x$ 试图最小化 $f(x)$，但被一根“弹簧”拉向 $z$（加上偏差 $u$）。这通常涉及矩阵求逆或梯度下降。
            
        2.  **z-update (本地优化 B / 近端算子)**：
            $$ z^{k+1} := \text{argmin}_z \left( g(z) + \frac{\rho}{2} \|x^{k+1} - z + u^k\|_2^2 \right) $$
            > $z$ 试图最小化 $g(z)$，同时也被拉向 $x$。对于 L1 范数等非平滑函数，这一步通常有解析解（如**软阈值**），非常高效。
            
        3.  **u-update (对偶变量更新 / 价格调整)**：
            $$ u^{k+1} := u^k + (x^{k+1} - z^{k+1}) $$
            > $u$ 是**对偶变量**（Dual Variable）。如果 $x$ 和 $z$ 还没重合，差异就会累积到 $u$ 中。$u$ 就像一个不断上涨的“罚款”，迫使 $x$ 和 $z$ 在下一轮必须靠得更近。
        
        ---
        ### 3. 本章具体应用：稀疏成像
        虽然 ADMM 是通用的，但在本章的 ISAR 成像中，我们令：
        *   $f(x) = \frac{1}{2}\|Ax-y\|_2^2$ (数据保真项)
        *   $g(z) = \lambda \|z\|_1$ (稀疏约束项)
        
        此时 ADMM 退化为经典的 LASSO 求解器。
        """)
        
        st.divider()
        st.subheader("🛠️ 交互式实验室：ADMM 的几何直觉")
        
        col_demo_ctrl, col_demo_plot = st.columns([1, 2.5])

        with col_demo_ctrl:
            st.markdown("##### 🎛️ 求解器参数")
            rho_demo = st.slider("惩罚参数 Rho ($\rho$)", 0.1, 5.0, 1.0, 0.1, 
                               help="控制一致性约束的强度。Rho 越大，x 和 z 被强行拉近的速度越快，但可能导致优化路径僵硬。")
            
            st.info(f"""
            **当前设定 $\\rho = {rho_demo}$**
            
            **图示说明**:
            此演示模拟了一个通用的二维优化问题。
            * 🟢 **绿色点 ($x$)**: 试图跑向数据中心。
            * 🔴 **红色虚线 ($z$)**: 试图满足某种约束（如被截断）。
            * 🔵 **蓝色力 ($u$)**: 累积误差产生的“牵引力”。
            
            ADMM 的魅力在于：即使 $f(x)$ 和 $g(z)$ 性质截然不同，也能通过 $u$ 的协调最终收敛。
            """)

        with col_demo_plot:
            # --- 通用优化路径模拟 ---
            steps = 20
            # 初始状态
            x_curr = np.array([4.0, 4.0])
            z_curr = np.array([4.0, 4.0])
            u_curr = np.zeros(2)
            
            path_x = [x_curr]
            
            # 模拟简单的 ADMM 动力学
            # 假设 f(x) 想要 x 去 (0,0)
            # 假设 g(z) 想要 z 稀疏 (靠近轴)
            for k in range(steps):
                # 1. x-update: 最小二乘代理 (Quadratic Prox)
                # min 1/2*||x||^2 + rho/2*||x - z + u||^2
                # 导数: x + rho(x - z + u) = 0 => x(1+rho) = rho(z-u)
                x_next = (rho_demo * (z_curr - u_curr)) / (1.0 + rho_demo)
                
                # 2. z-update: L1 范数代理 (Soft Thresholding)
                # min lambda*||z||_1 + rho/2*||x - z + u||^2
                # v = x + u
                v = x_next + u_curr
                lam = 0.5 # 模拟的正则化参数
                threshold = lam / rho_demo 
                z_next = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
                
                # 3. u-update
                u_next = u_curr + (x_next - z_next)
                
                x_curr, z_curr, u_curr = x_next, z_next, u_next
                path_x.append(x_curr)
            
            path_x = np.array(path_x)

            fig_dyn, ax_dyn = plt.subplots(figsize=(8, 5))
            fig_dyn.patch.set_facecolor('#ffffff')
            
            # 绘制等高线背景 (模拟 f(x) 的地形)
            delta = 0.1
            x_range = np.arange(-1.5, 5.0, delta)
            y_range = np.arange(-1.5, 5.0, delta)
            X, Y = np.meshgrid(x_range, y_range)
            Z = X**2 + Y**2 # 简单的凸函数
            ax_dyn.contour(X, Y, Z, levels=10, colors='gray', alpha=0.2)
            
            # 绘制路径
            ax_dyn.plot(path_x[:,0], path_x[:,1], 'o-', color='#2E7D32', markersize=6, linewidth=2, label='变量 X 的迭代轨迹')
            ax_dyn.plot(path_x[0,0], path_x[0,1], 'kD', label='起点')
            ax_dyn.plot(0, 0, 'r*', markersize=15, label='全局最优解')
            
            # 动态标题
            if rho_demo < 0.5:
                status_text = "约束过松 (Under-penalized): 轨迹偏离，收敛震荡"
                title_color = 'red'
            elif rho_demo > 3.0:
                status_text = "约束过紧 (Over-penalized): 步长受限，移动缓慢"
                title_color = 'orange'
            else:
                status_text = "参数均衡 (Balanced): 轨迹平滑，快速收敛"
                title_color = 'green'
                
            ax_dyn.set_title(f"Rho={rho_demo} | {status_text}", color=title_color, fontsize=11)
            ax_dyn.set_xlabel("参数空间 X1")
            ax_dyn.set_ylabel("参数空间 X2")
            ax_dyn.set_xlim(-1, 5)
            ax_dyn.set_ylim(-1, 5)
            ax_dyn.legend()
            ax_dyn.grid(True, alpha=0.3)
            
            st.pyplot(fig_dyn)
            plt.close(fig_dyn)

    # ==========================================
    # Tab 2: 仿真运行 (ISAR 实例)
    # ==========================================
    with tab_sim:
        st.markdown("在此页面，我们将 ADMM 算法的具体实现（LASSO 形式）应用于 **ISAR 稀疏成像** 问题。")
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 实验参数")
            
            st.markdown("**1. 场景设置**")
            sampling_rate = st.slider("随机降采样率", 0.1, 0.8, 0.3, step=0.05, key="admm_sr")
            
            st.markdown("**2. ADMM 求解器**")
            lambda_ratio = st.slider("稀疏正则 (Lambda)", 0.01, 0.2, 0.04, step=0.01, key="admm_lam",
                                   help="对应原理中的 g(z) 强度。值越大，图像越稀疏。")
            rho = st.slider("惩罚参数 (Rho)", 0.1, 5.0, 1.5, step=0.1, key="admm_rho",
                           help="对应原理中的步骤步长。")
            max_iter = st.slider("最大迭代次数", 10, 200, 50, step=10, key="admm_iter")
            
            st.divider()
            run_btn = st.button("🚀 运行 ADMM 重建", type="primary")
            
        # 数据准备
        raw, rd_img, _, _ = get_isar_data()
        N_pulses = raw.shape[1]
        np.random.seed(42)
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :]
        
        img_fft = compute_rd_image(raw_sparse)
        img_fft = np.flipud(img_fft) 

        with col_main:
            if run_btn:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                solver = CS_ISAR_Solver(raw_sparse, mask, max_iter=max_iter)
                
                start_time = time.time()
                progress_text.text("🔄 ADMM 迭代求解中 (Operator Splitting)...")
                
                # 安全调用
                if hasattr(solver, 'admm'):
                    img_admm = solver.admm(rho=rho, lambda_ratio=lambda_ratio)
                else:
                    st.error("❌ 错误：Utils 库中未找到 'admm' 方法。")
                    img_admm = np.zeros_like(img_fft)
                
                img_admm = np.flipud(img_admm)
                duration = end_time = time.time() - start_time
                
                progress_bar.progress(100)
                progress_text.empty()
                
                # 绘图
                fig_res = plot_isar_comparison(rd_img, img_fft, img_admm, "ADMM", duration, sampling_rate)
                st.pyplot(fig_res)
                plt.close(fig_res)
                
                # 保存结果
                st.session_state['admm_result'] = {
                    'image': img_admm,
                    'time': duration,
                    'rho': rho,
                    'loss': getattr(solver, 'loss_history', [])
                }
                
                st.success(f"✅ ADMM 重建完成！耗时: {duration:.3f} 秒")
                
            else:
                st.info("👈 点击左侧按钮，将通用的 ADMM 算法应用于具体的雷达数据。")
                fig_placeholder = plot_isar_comparison(rd_img, img_fft, np.zeros_like(rd_img), "待运行", 0, sampling_rate)
                st.pyplot(fig_placeholder)
                plt.close(fig_placeholder)

    # ==========================================
    # Tab 3: 稳定性分析
    # ==========================================
    with tab_analysis:
        if 'admm_result' in st.session_state:
            res = st.session_state['admm_result']
            losses = res.get('loss', [])
            
            st.markdown("### 🔍 ADMM 收敛特性分析")
            
            col_an_text, col_an_plot = st.columns([1, 2])
            
            with col_an_text:
                st.markdown(f"""
                **本次运行统计**:
                * **Rho 参数**: {res['rho']}
                * **总耗时**: {res['time']:.4f} s
                
                **为什么 ADMM 的 Loss 曲线很特别？**
                不同于梯度下降法的平滑单调下降，ADMM 的曲线往往呈现出**“多阶段”**特征：
                
                1.  **搜寻期**: $x$ 和 $z$ 差异较大，对偶变量 $u$ 快速调整，Loss 可能震荡。
                2.  **锁定期**: 随着 $x \\approx z$，算法进入线性收敛阶段，Loss 像下台阶一样下降。
                
                这种“先全局调整，后局部精修”的特性，使得 ADMM 在很多非凸或非平滑问题上比纯梯度方法更鲁棒。
                """)
                
            with col_an_plot:
                if losses and len(losses) > 0:
                    fig_loss, ax_loss = styler.get_fig_ax("Lagrangian Loss Curve")
                    ax_loss.plot(losses, 'g-', linewidth=2, label='ADMM Loss')
                    ax_loss.set_xlabel("Iterations")
                    ax_loss.set_ylabel("Loss (Log Scale)")
                    ax_loss.set_yscale('log')
                    ax_loss.set_title("ADMM 迭代收敛过程")
                    ax_loss.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax_loss.legend()
                    st.pyplot(fig_loss)
                    plt.close(fig_loss)
                else:
                    st.info("暂无 Loss 数据。")
        else:
            st.warning("⚠️ 请先在“仿真运行”标签页运行实验。")

if __name__ == "__main__":
    render()