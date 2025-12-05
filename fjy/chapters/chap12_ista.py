import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import styler, get_isar_data, compute_rd_image, CS_ISAR_Solver, plot_isar_comparison

def render():
    # --- 全局配置 ---
    st.markdown("""
    <style>
    .main-text {font-size:16px; line-height:1.6;}
    .math-box {background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #ff4b4b;}
    </style>
    """, unsafe_allow_html=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    tab_theory, tab_sim, tab_analysis = st.tabs(["📖 核心原理深度剖析", "🔬 ISAR 仿真实验室", "📉 收敛性能诊断"])

    # ==========================================
    # Tab 1: 理论深度解析 (含几何直观与算子交互)
    # ==========================================
    with tab_theory:
        st.markdown("### 1. 从数学公式到物理意义")
        st.markdown("""
        <div class="main-text">
        ISTA (Iterative Shrinkage-Thresholding Algorithm) 是解决线性逆问题的经典算法。
        在雷达成像中，我们的目标是从少量的回波数据y中恢复出清晰的图像x。
        这是一个典型的 <b>LASSO (Least Absolute Shrinkage and Selection Operator)</b> 问题：
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"""
        \min_{\mathbf{x}} \underbrace{\frac{1}{2} \|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2}_{\text{数据保真项 (Data Fidelity)}} + \underbrace{\lambda \|\mathbf{x}\|_1}_{\text{稀疏正则项 (Sparsity Prior)}}
        """)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info("""
            **项 1：数据保真**
            * **物理含义**：重建的图像投影回去，必须和接收到的雷达回波一致。
            * **处理手段**：梯度下降 (Gradient Descent)。
            * **直观理解**：尽量靠近数据的“圆心”。
            """)
        with col_t2:
            st.success("""
            **项 2：稀疏正则**
            * **物理含义**：假设背景是纯黑的，只有少量的强散射点（飞机、舰船）。
            * **处理手段**：软阈值 (Soft Thresholding)。
            * **直观理解**：尽量让非零像素少，去除噪声。
            """)

        st.markdown("### 2. 迭代二部曲 (详细拆解)")
        st.markdown("""
        <div class="main-text">
        从最优化理论的角度，ISTA 是在交替优化目标函数的两个部分：<b>数据保真项</b>（最小二乘误差）和<b>稀疏正则项</b>（L1 范数）。
        </div>
        """, unsafe_allow_html=True)

        # --- 第一步：梯度下降 ---
        st.markdown("#### 🟢 第一步：梯度下降 (数据保真 / Data Fidelity)")
        st.markdown("这一步旨在最小化平滑的最小二乘误差项")

        st.latex(r"""
        \mathbf{z}^{(k)} = \mathbf{x}^{(k)} - \alpha \nabla f(\mathbf{x}^{(k)}) 
        = \mathbf{x}^{(k)} - \alpha \mathbf{A}^H (\mathbf{A}\mathbf{x}^{(k)} - \mathbf{y})
        """)

        st.info("""
        **数学原理**：
        这是一个标准的**梯度下降 (Gradient Descent)** 步骤。
        *   **目标**：确保重建图像 $\mathbf{x}$ 经过正向模型 $\mathbf{A}$ 投影后，与观测数据 $\mathbf{y}$ 尽可能一致（即满足**数据一致性**）。
        *   **机制**：计算残差向量 $\mathbf{r} = \mathbf{A}\mathbf{x} - \mathbf{y}$，并通过伴随算子（Adjoint Operator, $\mathbf{A}^H$）将其从测量域映射回图像域。这代表了目标函数在当前点的下降方向。
        """)

        # --- 第二步：软阈值 ---
        st.markdown("#### 🔴 第二步：软阈值 (稀疏正则 / L1 Regularization)")
        st.markdown("这一步旨在处理非平滑的正则化项 $g(\mathbf{x}) = \lambda \|\mathbf{x}\|_1$，实际上是计算 L1 范数的**近端算子 (Proximal Operator)**。")

        st.latex(r"""
        \mathbf{x}^{(k+1)} = \text{prox}_{\lambda \alpha g}(\mathbf{z}^{(k)}) = \text{soft}\left( \mathbf{z}^{(k)}, \lambda \alpha \right)
        """)

        st.error("""
        **数学原理**：
        这是一个**近端映射 (Proximal Mapping)** 步骤，也是 L1 正则化问题的解析解。
        *   **目标**：在图像域引入**稀疏先验 (Sparsity Prior)**，即假设目标场景由少量强散射点组成。
        *   **机制**：软阈值算子是非线性的收缩函数。它不仅将幅值小于阈值 $\lambda$ $ \alpha$ 的噪声分量置零（实现稀疏性），同时将大于阈值的信号分量向零收缩（这解释了 LASSO 的有偏估计特性）。
        """)

        st.caption("🔄 **收敛逻辑**：算法在“降低数据拟合误差”与“满足稀疏约束”这两个目标之间交替投影，最终收敛到目标泛函的全局最优解。")
        st.divider()
        st.markdown("### 2. 核心机制：为什么 L1 能带来稀疏？")
        
        # --- 交互演示 1: 软阈值算子 ---
        st.markdown("#### 🧪 实验 A：软阈值算子 (Soft Thresholding Operator)")
        st.markdown("ISTA 的核心在于每一步梯度下降后，都调用一次软阈值函数。这相当于一个**非线性滤波器**。")

        def soft_threshold(x, thresh):
            return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

        col_demo_ctrl, col_demo_viz = st.columns([1, 2])
        
        with col_demo_ctrl:
            st.markdown("**参数控制台**")
            thresh = st.slider("截断阈值 (T)", 0.0, 3.0, 1.0, 0.1, help="任何幅值小于 T 的信号都将被强制归零。")
            noise_std = st.slider("输入噪声强度", 0.0, 1.0, 0.2, 0.05)
            
            st.markdown("""
            **观察指南**：
            1. **死区 (Dead Zone)**：注意看右图中红色背景区域，落入此区域的输入全部变为 0。这就是稀疏性的来源。
            2. **收缩效应**：注意看大于 T 的信号，输出比输入小了 T。这代表能量的收缩。
            """)

        with col_demo_viz:
            # 动态生成数据
            x = np.linspace(-4, 4, 500)
            y = soft_threshold(x, thresh)
            
            # 创建双子图：函数映射 + 信号演示
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # 子图 1: 映射函数
            ax1.plot(x, x, 'k--', alpha=0.3, label='y=x (Identity)')
            ax1.plot(x, y, 'b-', linewidth=2, label='Soft Threshold')
            ax1.fill_between([-thresh, thresh], -4, 4, color='red', alpha=0.1, label='Dead Zone (Zeroed)')
            ax1.set_title(f"映射函数 T={thresh}")
            ax1.set_xlabel("Input (Noisy)")
            ax1.set_ylabel("Output (Sparse)")
            ax1.grid(True, linestyle=':')
            ax1.legend()
            
            # 子图 2: 实际信号处理
            np.random.seed(42)
            t = np.arange(50)
            sig_clean = np.zeros(50)
            sig_clean[10] = 3.5
            sig_clean[25] = -2.0
            sig_clean[40] = 1.5
            sig_noisy = sig_clean + np.random.randn(50) * noise_std
            sig_recon = soft_threshold(sig_noisy, thresh)
            
            ax2.stem(t, sig_noisy, linefmt='gray', markerfmt='k.', basefmt=' ', label='Noisy Input')
            ax2.stem(t, sig_recon, linefmt='b-', markerfmt='bo', basefmt=' ', label='Thresholded')
            ax2.axhline(thresh, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(-thresh, color='r', linestyle='--', alpha=0.5)
            ax2.set_title("1D 信号去噪效果")
            ax2.legend()
            
            st.pyplot(fig)
            
            # 统计非零元素
            nnz_input = np.count_nonzero(np.abs(sig_noisy) > 1e-3)
            nnz_output = np.count_nonzero(np.abs(sig_recon) > 1e-3)
            st.caption(f"📊 稀疏度变化: 输入非零点 {nnz_input} -> 输出非零点 **{nnz_output}** (目标值: 3)")

    # ==========================================
    # Tab 2: 仿真运行
    # ==========================================
    with tab_sim:
        st.markdown("### 🚀 ISTA 算法实战：ISAR 稀疏成像")
        st.markdown("在这里，我们将模拟真实雷达场景：雷达发射脉冲随机丢失（稀疏采样），我们需要用 ISTA 恢复出目标。")
        
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 实验参数设置")
            
            with st.expander("📡 采样设置 (物理层)", expanded=True):
                sampling_rate = st.slider("降采样率 (SR)", 0.1, 0.8, 0.4, 0.05, 
                                        help="仅保留多少比例的雷达回波。越低越难重建。")
            
            with st.expander("🧠 算法设置 (求解器)", expanded=True):
                lambda_ratio = st.slider("稀疏正则 (Lambda)", 0.01, 0.20, 0.05, 0.01,
                                       help="控制阈值 T 的大小。太大会抹除目标，太小会保留噪声。")
                alpha = st.slider("梯度步长 (Alpha)", 0.1, 2.0, 1.0, 0.1,
                                help="梯度下降的步长。")
                max_iter = st.number_input("最大迭代次数", 20, 500, 100, 20)
            
            run_btn = st.button("开始重建 (Run ISTA)", type="primary")

        # 数据准备
        raw, rd_img, _, _ = get_isar_data()
        N_pulses = raw.shape[1]
        np.random.seed(123) # 固定随机数以便复现
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :] 
        
        # 传统 RD 结果（基准）
        img_fft = compute_rd_image(raw_sparse)
        img_fft = np.flipud(img_fft) 

        with col_main:
            if run_btn:
                # 进度显示
                progress_bar = st.progress(0)
                status = st.empty()
                
                # 初始化求解器
                solver = CS_ISAR_Solver(raw_sparse, mask, max_iter=max_iter)
                
                # 记录时间
                t_start = time.time()
                
                # --- 核心算法调用 ---
                # 为了演示详细过程，这里实际上调用的是 solver.ista
                img_ista = solver.ista(lambda_ratio=lambda_ratio, alpha=alpha)
                # -------------------
                
                t_end = time.time()
                duration = t_end - t_start
                img_ista = np.flipud(img_ista)
                
                progress_bar.progress(100)
                status.success(f"✅ 计算完成！耗时: {duration:.3f} 秒")
                
                # 绘图对比
                fig_res = plot_isar_comparison(rd_img, img_fft, img_ista, "ISTA Reconstructed", duration, sampling_rate)
                st.pyplot(fig_res)
                
                # 保存运行状态供 Tab 3 分析
                st.session_state['ista_run'] = {
                    'loss_history': getattr(solver, 'loss_history', []),
                    'final_img': img_ista,
                    'params': {'lambda': lambda_ratio, 'iter': max_iter, 'sr': sampling_rate}
                }
                
                # 详细结果解释
                st.info(f"""
                **结果解读**：
                1. **图1 (Ground Truth)**：使用 100% 数据的理想成像。
                2. **图2 (Sparse FFT)**：仅使用 {sampling_rate*100}% 数据直接 FFT。由于数据缺失，导致严重的**旁瓣干扰和栅瓣伪影**，目标被淹没。
                3. **图3 (ISTA)**：利用稀疏先验，ISTA 成功抑制了背景伪影，恢复了飞机的轮廓点。
                """)

            else:
                st.info("👈 请调整左侧参数并运行仿真。")
                fig_holder = plot_isar_comparison(rd_img, img_fft, np.zeros_like(rd_img), "Waiting...", 0, sampling_rate)
                st.pyplot(fig_holder)

    # ==========================================
    # Tab 3: 收敛分析
    # ==========================================
    with tab_analysis:
        st.markdown("### 📈 算法收敛性诊断")
        
        if 'ista_run' in st.session_state:
            run_data = st.session_state['ista_run']
            loss_hist = run_data['loss_history']
            
            if len(loss_hist) > 0:
                col_an1, col_an2 = st.columns([2, 1])
                
                with col_an1:
                    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                    ax_loss.plot(loss_hist, 'r-', linewidth=2, label='Objective Function Value')
                    ax_loss.set_title("ISTA Loss Convergence Curve")
                    ax_loss.set_xlabel("Iteration (k)")
                    ax_loss.set_ylabel("Loss: $1/2||Ax-y||^2 + \lambda||x||_1$")
                    ax_loss.set_yscale('log')
                    ax_loss.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax_loss.legend()
                    st.pyplot(fig_loss)
                    
                with col_an2:
                    st.markdown("#### 诊断报告")
                    init_loss = loss_hist[0]
                    final_loss = loss_hist[-1]
                    drop_pct = (init_loss - final_loss) / init_loss * 100
                    
                    st.metric("初始 Loss", f"{init_loss:.2e}")
                    st.metric("最终 Loss", f"{final_loss:.2e}")
                    st.metric("下降幅度", f"{drop_pct:.1f}%")
                    
                    st.markdown("""
                    **曲线特征分析**：
                    * **阶段 I (快速下降)**：梯度下降主导，快速拟合观测数据。
                    * **阶段 II (拖尾效应)**：软阈值算子在微调小系数，收敛变得非常缓慢。
                    """)
                    
                st.warning("""
                **⚠️ 性能瓶颈提示**：
                ISTA 的收敛速率仅为 $O(1/k)$。在要求高精度的场景下，它需要非常多的迭代次数。
                
                **改进方案**：
                可以使用 **FISTA (Fast ISTA)**，引入动量项（Momentum），将收敛速率提升至 $O(1/k^2)$。
                """)
            else:
                st.warning("未能获取 Loss 历史数据，请检查求解器实现。")
        else:
            st.info("请先在 [仿真实验] 标签页运行一次算法。")

if __name__ == "__main__":
    render()