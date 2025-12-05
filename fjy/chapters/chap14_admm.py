import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
# 假设 utils 库依然存在，保持原有引用
from utils import styler, get_isar_data, compute_rd_image, CS_ISAR_Solver, plot_isar_comparison

def render():
    # --- 全局配置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    st.header("💡 深入解析：ADMM 交替方向乘子法")
    
    # Tab 分层结构
    tab_theory, tab_sim, tab_analysis = st.tabs(["📖 核心原理深度拆解", "🚀 ISAR 仿真实验", "📊 收敛性分析"])

    # ==========================================
    # Tab 1: 算法原理 (深度重写 - 增加推导过程)
    # ==========================================
    with tab_theory:
        st.markdown(r"""
        在稀疏成像（如 LASSO 问题）中，我们可以把要解决的目标函数包含两个截然不同的部分：
        
        $$ \min_x \underbrace{\frac{1}{2} \| Ax - y \|_2^2}_{f(x): \text{数据保真 (光滑)}} + \underbrace{\lambda \| x \|_1}_{g(x): \text{稀疏约束 (非光滑)}} $$
        
        ADMM 的策略是 **“变量分裂 (Variable Splitting)”**：引入辅助变量 $z$，将问题转化为约束优化问题：
                    
        $$ \min_{x,z} f(x) + g(z) $$
                    
        $$ \text{s.t.} \quad x - z = 0 $$
        """)
        
        st.divider()
        
        st.markdown(r"""
        ### 2. 数学构造：从“标准形式”到“缩放形式”
        为了求解上述约束问题，我们首先写出 **增广拉格朗日函数 (Standard Augmented Lagrangian)**。
                    
        引入对偶变量（拉格朗日乘子） $\alpha$（注意：这里用 $\alpha$ 表示未缩放的乘子）：
                    
        $$ L_\rho(x, z, \alpha) = f(x) + g(z) + \alpha^T(x - z) + \frac{\rho}{2}\|x - z\|_2^2 $$
                    
        这里 $\frac{\rho}{2}\|x - z\|_2^2$ 是惩罚项，用于增强凸性和收敛性。
                    
        """)

        # --- 推导核心部分 ---
        st.info("👇 点击下方展开，查看“缩放形式”是如何推导出来的（数学推导）")
        with st.expander("🧮 详细推导过程：配方技巧"):
            st.markdown(r"""
            **第一步：写出标准增广拉格朗日函数**
            
            引入对偶变量（拉格朗日乘子） $\alpha$（注意：这里用 $\alpha$ 表示未缩放的乘子）：
            $$ L_\rho(x, z, \alpha) = f(x) + g(z) + \alpha^T(x - z) + \frac{\rho}{2}\|x - z\|_2^2 $$
            这里 $\frac{\rho}{2}\|x - z\|_2^2$ 是惩罚项，用于增强凸性和收敛性。

            ---
            
            **第二步：合并最后两项（配方）**
            
            我们需要处理线性项 $\alpha^T(x-z)$ 和二次项 $\frac{\rho}{2}\|x-z\|^2$。为了简化，我们尝试将它们合并成一个平方项。
            
            回顾完全平方公式 $\|a+b\|^2 = \|a\|^2 + 2a^Tb + \|b\|^2$，我们可以进行如下变换：
            
            $$
            \begin{aligned}
            \alpha^T(x - z) + \frac{\rho}{2}\|x - z\|_2^2 &= \frac{\rho}{2} \left( \frac{2}{\rho}\alpha^T(x - z) + \|x - z\|_2^2 \right) \\
            &= \frac{\rho}{2} \left( \|x - z\|_2^2 + 2(x - z)^T \left( \frac{\alpha}{\rho} \right) \right)
            \end{aligned}
            $$
            
            为了凑成完全平方，我们加上并减去常数项 $\|\frac{\alpha}{\rho}\|_2^2$：
            
            $$
            \begin{aligned}
            &= \frac{\rho}{2} \left( \underbrace{\|x - z\|_2^2 + 2(x - z)^T \left( \frac{\alpha}{\rho} \right) + \left\| \frac{\alpha}{\rho} \right\|_2^2}_{\text{是一个完全平方}} - \left\| \frac{\alpha}{\rho} \right\|_2^2 \right) \\
            &= \frac{\rho}{2} \left\| x - z + \frac{\alpha}{\rho} \right\|_2^2 - \frac{\rho}{2} \left\| \frac{\alpha}{\rho} \right\|_2^2
            \end{aligned}
            $$

            ---
            
            **第三步：变量代换（Scaling）**
            
            定义 **缩放对偶变量 (Scaled Dual Variable)** $u$ 为：
            $$ u = \frac{1}{\rho} \alpha $$
            
            代入上式，原来的拉格朗日函数就变成了极其简洁的 **缩放形式**：
            $$ L_\rho(x, z, u) = f(x) + g(z) + \frac{\rho}{2} \|x - z + u\|_2^2 - \frac{\rho}{2}\|u\|_2^2 $$
            
            **为什么这样做？**
            这样做的好处是，在更新 $x$ 和 $z$ 时，目标函数看起来仅仅是在最小化一个简单的欧氏距离（加上正则项），这非常利于编程实现。
            """)

        st.markdown(r"""
        最终我们使用的公式为：
        $$ L_\rho(x, z, u) = \frac{1}{2}\|Ax - y\|_2^2 + \lambda\|z\|_1 + \frac{\rho}{2} \|x - z + u\|_2^2 - \text{const} $$
        
        其中：
        *   $u$：**缩放对偶变量**，代表累积误差。
        *   $\rho$：惩罚参数，控制对 $x-z$ 误差的容忍度。
                    
        #### 💡 为什么要用缩放形式？
        如果不使用缩放形式，我们的二次项是 $\alpha^T(x-z) + \frac{\rho}{2}\|x-z\|^2$。这在求导时会产生交叉项，处理起来很麻烦。
        
        使用缩放形式后，我们将线性项和二次项合并成了一个完美的**欧氏距离平方项** $\frac{\rho}{2}\|x - z + u\|^2$。
        这使得子问题变成了我们非常熟悉的去噪或投影问题：
        *   **$x$-update**：变成了标准的最小二乘问题（Ridge Regression 形式）。
        *   **$z$-update**：变成了标准的 Proximal Operator 问题（直接套用软阈值公式）。
                    
        """)
        
        st.divider()

        st.markdown(r"""
        ### 3. 迭代三部曲
        ADMM 将联合优化问题拆分为三个简单的子问题，交替求解：
        
        #### 🟢 第一步：x-update (模型反演)
        $$ x^{k+1} = \arg\min_x \left( \frac{1}{2}\|Ax - y\|_2^2 + \frac{\rho}{2}\|x - z^k + u^k\|_2^2 \right) $$
        
        #### 🔴 第二步：z-update (稀疏去噪)
        利用缩放形式，这里变成了一个标准的**近端算子 (Proximal Operator)** 问题：
        $$ z^{k+1} = \arg\min_z \left( \lambda\|z\|_1 + \frac{\rho}{2}\| (x^{k+1} + u^k) - z \|_2^2 \right) $$
        
        这个形式直观地告诉我们：**我们要找一个 $z$，让它既稀疏（由 $\lambda$ 控制），又尽量靠近 $x+u$。**
        这有闭式解——**软阈值 (Soft Thresholding)**：
        $$ z^{k+1} = \text{soft}\left( x^{k+1} + u^k, \frac{\lambda}{\rho} \right) $$
        
        #### 🔵 第三步：u-update (对偶变量更新)
        对应于原始对偶变量更新 $\alpha^{k+1} = \alpha^k + \rho(x^{k+1} - z^{k+1})$，除去 $\rho$ 后：
        $$ u^{k+1} = u^k + (x^{k+1} - z^{k+1}) $$
                    
        > **🤔 为什么要这么更新？**
        > *   **本质**：这是在对偶问题上做梯度上升。
        > *   **直观理解**：$(x - z)$ 是当前的约束违反程度（残差）。
        >     *   如果 $x > z$，说明 $x$ 跑偏了，我们就增加 $u$，在下一次 $x$-update 中把 $x$ 拉回来。
        >     *   $u$ 会不断累积历史误差，直到 $x$ 和 $z$ 完全一致 ($x-z=0$)，此时 $u$ 停止变化，算法收敛。
        """)
        
        st.divider()
        st.subheader("🧪 交互演示：2D 空间中的 ADMM 轨迹")
        
        col_demo_ctrl, col_demo_plot = st.columns([1, 2])

        with col_demo_ctrl:
            st.info("此演示模拟一个简单的二维去噪问题：目标是找到原点 (0,0)，但受 L1 约束影响。")
            rho_demo = st.slider("惩罚参数 Rho ($\rho$)", 0.1, 5.0, 1.0, 0.1)
            lam_demo = st.slider("正则化强度 Lambda ($\lambda$)", 0.1, 2.0, 0.5, 0.1)
            st.caption(f"阈值 Threshold = $\lambda / \\rho$ = {lam_demo/rho_demo:.2f}")
            
            st.markdown(r"""
            **参数影响说明**：
            *   **Rho ($\rho$)**：控制 $x$ 和 $z$ 必须多快达成一致。
                *   **数学原理**：在 $x$-update 中，$\rho$ 是二次惩罚项 $\frac{\rho}{2}\|x - z + u\|_2^2$ 的权重。
                *   **$\rho$ 越大**：惩罚项占比越重，$x$ 被强力拉向 $z$，收敛步长变小但更稳（轨迹更直）。
                *   **$\rho$ 越小**：允许 $x$ 和 $z$ 暂时分离，可能产生螺旋震荡。
            *   **Lambda ($\lambda$)**：控制稀疏度（阈值大小）。
                *   **数学原理**：在 $z$-update 的软阈值公式 $z = \text{soft}(x+u, \lambda/\rho)$ 中，$\lambda$ 直接决定截断阈值。
                *   **$\lambda$ 越大**：阈值 $\frac{\lambda}{\rho}$ 越高，更多的小幅值分量被置零，结果越趋向于 0（更稀疏）。
            """)


        with col_demo_plot:
            # --- 模拟 ADMM 求解 min 1/2||x||^2 + lam||z||_1 ---
            # 这是一个简单的去噪模型，目标是 (0,0)
            steps = 15
            x_path, z_path = [], []
            
            # 初始点 (故意设远一点)
            x = np.array([3.5, 2.5])
            z = np.array([3.5, 2.5])
            u = np.array([0.0, 0.0])
            
            x_path.append(x)
            z_path.append(z)
            
            for k in range(steps):
                # 1. x-update: min 1/2||x||^2 + rho/2||x - v||^2  (v = z-u)
                # 导数: x + rho(x - z + u) = 0  => x(1+rho) = rho(z - u)
                x = (rho_demo * (z - u)) / (1 + rho_demo)
                
                # 2. z-update: Soft Thresholding
                # input is (x + u)
                val = x + u
                threshold = lam_demo / rho_demo
                # 软阈值函数
                z = np.sign(val) * np.maximum(np.abs(val) - threshold, 0)
                
                # 3. u-update
                u = u + (x - z)
                
                x_path.append(x)
                z_path.append(z)
            
            x_path = np.array(x_path)
            z_path = np.array(z_path)
            
            # 绘图
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 背景等高线 (代表 f(x) = 1/2||x||^2)
            grid = np.linspace(-1, 4, 100)
            X, Y = np.meshgrid(grid, grid)
            Z = 0.5 * (X**2 + Y**2)
            ax.contour(X, Y, Z, levels=15, cmap='Greys', alpha=0.3)
            
            # 绘制轨迹
            ax.plot(x_path[:,0], x_path[:,1], 'o-', color='#1f77b4', label='x (物理变量)', markersize=5)
            ax.plot(z_path[:,0], z_path[:,1], 'x--', color='#d62728', label='z (稀疏变量)', markersize=5)
            
            # 标注
            ax.plot(0, 0, 'k*', markersize=15, label='全局最优 (0,0)')
            ax.plot(x_path[0,0], x_path[0,1], 'kD', label='起点')
            
            ax.set_title(f"ADMM 迭代轨迹 (Rho={rho_demo}, Lam={lam_demo})")
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlim(-1, 4)
            ax.set_ylim(-1, 4)
            
            st.pyplot(fig)
            st.caption("""
            * **蓝色实线 (x)**：试图走向圆心（最小二乘解），受 $u$ 牵引。
            * **红色虚线 (z)**：总是试图落在坐标轴上（稀疏解），这是软阈值的作用。
            * **收敛**：随着 $u$ 的调整，红蓝两点最终会重合。
            """)

    # ==========================================
    # Tab 2: 仿真运行 (保持原样)
    # ==========================================
    with tab_sim:
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 实验设置")
            sampling_rate = st.slider("降采样率 (保留比例)", 0.1, 0.8, 0.4, step=0.05)
            
            st.markdown("---")
            st.subheader("ADMM 参数")
            lambda_ratio = st.slider("正则化 Lambda", 0.01, 0.2, 0.05, step=0.01, help="控制稀疏度。值越大，去噪越强，但也可能丢失细节。")
            rho = st.slider("惩罚参数 Rho", 0.5, 5.0, 2.0, step=0.1, help="影响收敛速度。太小收敛慢，太大可能震荡。")
            max_iter = st.number_input("迭代次数", 10, 500, 50, step=10)
            
            run_btn = st.button("开始重建", type="primary")

        # 获取数据
        raw, rd_img, _, _ = get_isar_data()
        N_pulses = raw.shape[1]
        np.random.seed(42) # 固定随机种子以便复现
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :]
        
        # 传统 RD 成像（作为对比）
        img_fft = compute_rd_image(raw_sparse)
        img_fft = np.flipud(img_fft)

        with col_main:
            if run_btn:
                # 进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                solver = CS_ISAR_Solver(raw_sparse, mask, max_iter=max_iter)
                
                start_time = time.time()
                status_text.markdown("**正在迭代求解中...** `x-update` -> `z-update` -> `u-update`")
                
                # 调用求解器
                if hasattr(solver, 'admm'):
                    # 实际运行 ADMM
                    img_admm = solver.admm(rho=rho, lambda_ratio=lambda_ratio)
                else:
                    st.error("Utils 模块中未定义 admm 方法")
                    img_admm = np.zeros_like(img_fft)
                
                end_time = time.time()
                duration = end_time - start_time
                img_admm = np.flipud(img_admm)
                
                progress_bar.progress(100)
                status_text.success(f"✅ 重建完成！耗时: {duration:.3f}s")
                
                # 结果对比绘图
                fig_res = plot_isar_comparison(rd_img, img_fft, img_admm, "ADMM", duration, sampling_rate)
                st.pyplot(fig_res)
                
                # 保存 Loss 供 Tab 3 分析
                st.session_state['admm_logs'] = {
                    'loss': solver.loss_history,
                    'rho': rho,
                    'lambda': lambda_ratio
                }
            else:
                st.info("👈 请调整左侧参数并点击按钮运行 ADMM 算法。")
                # 显示原始数据占位
                fig_placeholder = plot_isar_comparison(rd_img, img_fft, np.zeros_like(rd_img), "待运行", 0, sampling_rate)
                st.pyplot(fig_placeholder)

    # ==========================================
    # Tab 3: 稳定性分析 (保持原样)
    # ==========================================
    with tab_analysis:
        st.subheader("📉 收敛性诊断")
        
        if 'admm_logs' in st.session_state:
            logs = st.session_state['admm_logs']
            loss_history = logs['loss']
            
            col_chart, col_explain = st.columns([2, 1])
            
            with col_chart:
                fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
                ax_loss.plot(loss_history, color='#2ca02c', linewidth=2)
                ax_loss.set_title("ADMM 原始残差收敛曲线 (Primal Residual)")
                ax_loss.set_xlabel("Iterations")
                ax_loss.set_ylabel("||x - z||_2")
                ax_loss.set_yscale('log')
                ax_loss.grid(True, which="both", ls="--", alpha=0.5)
                st.pyplot(fig_loss)
                
            with col_explain:
                st.markdown(f"""
                **当前参数状态**:
                * $\\rho = {logs['rho']}$
                * $\\lambda = {logs['lambda']}$
                
                **曲线解读**:
                * 纵轴显示的是 **原始残差 (Primal Residual)** $r = \|x - z\|_2$。
                * 这代表了物理变量 $x$ 和稀疏变量 $z$ 之间的不一致程度。
                * ADMM 收敛的标志是该残差趋近于 0。
                * 如果曲线呈现锯齿状震荡，说明 $\\rho$ 可能过大；如果下降极其缓慢，说明 $\\rho$ 可能过小。
                """)
        else:
            st.warning("暂无数据。请先在 'ISAR 仿真实验' 中运行算法。")

if __name__ == "__main__":
    render()