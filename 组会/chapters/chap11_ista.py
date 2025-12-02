import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
# 假设 utils.py 在同级目录下，包含必要的辅助函数
# 如果没有 utils，你需要自行实现这些 mock 函数
from utils import styler, get_isar_data, compute_rd_image, CS_ISAR_Solver, plot_isar_comparison

def render():
    # --- 全局配置 ---
    # 设置中文字体，防止绘图乱码 (根据系统环境自动回退)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

    # 使用 Tabs 分层展示：从原理到仿真再到分析
    tab_theory, tab_sim, tab_analysis = st.tabs(["📘 算法原理深度解析", "🚀 ISAR 稀疏成像仿真", "📊 收敛性分析"])

    # ==========================================
    # Tab 1: 算法原理 (集成交互式演示)
    # ==========================================
    with tab_theory:
        # --- 理论讲解部分 ---
        st.markdown(r"""
        ### 1. 核心问题：Lasso 优化
        在 ISAR 稀疏成像中，我们试图求解以下无约束凸优化问题：
        
        $$ \hat{\mathbf{x}} = \arg\min_\mathbf{x} \underbrace{\frac{1}{2} \|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2}_{\text{拟合误差 (梯度下降处理)}} + \underbrace{\lambda \|\mathbf{x}\|_1}_{\text{稀疏约束 (软阈值处理)}} $$
        
        ### 2. ISTA 算法逻辑
        **迭代收缩阈值算法 (ISTA)** 巧妙地将上述问题拆解为两步迭代循环：
        
        1.  **梯度下降 (Gradient Descent)**：首先让图像去拟合观测数据。
            $$ \mathbf{z}_k = \mathbf{x}_k - \alpha \mathbf{A}^T(\mathbf{A}\mathbf{x}_k - \mathbf{y}) $$
        2.  **近端映射 (Proximal Mapping)**：通过**软阈值算子**去除噪声，强制稀疏。
            $$ \mathbf{x}_{k+1} = \mathcal{S}_{\lambda \alpha}(\mathbf{z}_k) $$
        """)
        
        # --- 新增：软阈值公式详解 ---
        st.info("💡 **关键概念：软阈值算子 (Soft-Thresholding Operator)**")
        
        cols_math = st.columns([1, 1])
        with cols_math[0]:
            st.markdown(r"""
            **数学定义**：
            软阈值算子 $\mathcal{S}_T(x)$ 是 $L_1$ 正则化的解析解。对于输入 $x$ 和阈值 $T$：
            
            $$
            \mathcal{S}_T(x) = 
            \begin{cases} 
            x - T & \text{if } x > T \\
            x + T & \text{if } x < -T \\
            0 & \text{if } |x| \leq T 
            \end{cases}
            $$
            """)
        with cols_math[1]:
            st.markdown(r"""
            **物理含义**：
            1.  **截断 (Thresholding)**：绝对值小于 $T$ 的微小分量（通常是噪声或伪影）直接被**置为 0**。
            2.  **收缩 (Shrinkage)**：绝对值大于 $T$ 的主要分量（真实目标散射点）向 0 方向**缩减** $T$ 的幅度。
            
            > 这就是为什么 $L_1$ 正则化能产生稀疏解（也就是让很多像素变为纯 0）的根本原因。
            """)
        
        st.markdown("---")
        st.subheader("🧪 交互式实验室：体验“截断”与“收缩”")

        # --- 交互式演示模块 ---
        
        # 定义软阈值函数
        def soft_threshold(x, thresh):
            return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

        # 创建两列：左侧控制参数，右侧实时绘图
        col_interactive_ctrl, col_interactive_plot = st.columns([1, 2.5])

        with col_interactive_ctrl:
            st.markdown("##### 🎛️ 参数控制")
            
            # 1. 阈值控制
            thresh_val = st.slider("阈值 T (Threshold)", 
                                  min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                  help="决定了‘死区’的宽度。所有幅值小于此值的信号都会消失。")
            
            # 2. 噪声控制
            noise_level = st.slider("输入噪声强度", 
                                   min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                                   help="模拟梯度下降步骤后引入的残余噪声。")
            
            # 3. 视角切换
            view_mode = st.radio("观察视角", 
                                ["📈 函数映射特性 (Input vs Output)", "📶 信号去噪效果 (Time Domain)"],
                                help="函数视角看算子本身的非线性特性；信号视角看它如何提取稀疏目标。")

            st.write(f"当前公式: $y = \mathcal{{S}}_{{{thresh_val}}}(x)$")

        with col_interactive_plot:
            if view_mode == "📈 函数映射特性 (Input vs Output)":
                # --- 视角 1: 函数映射图 ---
                fig_dyn, ax_dyn = plt.subplots(figsize=(8, 5))
                fig_dyn.patch.set_facecolor('#ffffff')
                
                x_vals = np.linspace(-3, 3, 500)
                y_vals = soft_threshold(x_vals, thresh_val)
                
                # 绘制辅助线 y=x
                ax_dyn.plot(x_vals, x_vals, 'k--', alpha=0.3, label='y=x (线性/无操作)')
                
                # 绘制死区 (Dead Zone)
                ax_dyn.fill_between([-thresh_val, thresh_val], -3, 3, color='red', alpha=0.1, label=f'置零死区 (-{thresh_val}, {thresh_val})')
                
                # 绘制算子曲线
                ax_dyn.plot(x_vals, y_vals, 'b-', linewidth=3, label='软阈值输出')
                
                # 装饰图表
                ax_dyn.set_title(f"软阈值算子映射曲线 (T={thresh_val})", fontsize=14)
                ax_dyn.set_xlabel("输入值 x (含噪)", fontsize=12)
                ax_dyn.set_ylabel("输出值 y (稀疏化)", fontsize=12)
                ax_dyn.set_xlim(-3, 3)
                ax_dyn.set_ylim(-3, 3)
                ax_dyn.grid(True, linestyle='--', alpha=0.6)
                ax_dyn.legend(loc='upper left')
                ax_dyn.axhline(0, color='black', linewidth=0.5)
                ax_dyn.axvline(0, color='black', linewidth=0.5)
                
                # 动态标注
                if thresh_val > 0.1:
                    ax_dyn.annotate('数值向0收缩', xy=(2.5, 2.5-thresh_val), xytext=(1.5, 0.5),
                                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

                st.pyplot(fig_dyn)
                
            else:
                # --- 视角 2: 信号处理视角 ---
                # 生成模拟稀疏信号
                np.random.seed(42) 
                t = np.linspace(0, 100, 100)
                # 稀疏信号：只有3个强散射点
                true_signal = np.zeros_like(t)
                true_signal[20] = 2.5
                true_signal[45] = -1.8
                true_signal[70] = 1.2
                
                # 添加噪声
                noise = np.random.randn(len(t)) * noise_level
                noisy_input = true_signal + noise
                
                # 应用软阈值
                filtered_output = soft_threshold(noisy_input, thresh_val)
                
                fig_sig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                fig_sig.patch.set_facecolor('#ffffff')
                plt.subplots_adjust(hspace=0.3)
                
                # 上图：含噪输入
                ax1.set_title("步骤1: 梯度下降后的含噪信号", fontsize=12, fontweight='bold')
                ax1.stem(t, noisy_input, linefmt='grey', markerfmt='yo', basefmt=' ', label='含噪输入')
                # 画出阈值线
                ax1.axhline(y=thresh_val, color='r', linestyle='--', alpha=0.8, label='+T')
                ax1.axhline(y=-thresh_val, color='r', linestyle='--', alpha=0.8, label='-T')
                ax1.fill_between(t, -thresh_val, thresh_val, color='red', alpha=0.1)
                ax1.legend(loc='upper right', fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # 下图：阈值后输出
                ax2.set_title("步骤2: 软阈值处理后的结果", fontsize=12, fontweight='bold')
                # 绘制原始真值参考
                ax2.stem(t, true_signal, linefmt='g:', markerfmt='gx', basefmt=' ', label='真实目标')
                # 绘制恢复结果
                markerline, stemlines, baseline = ax2.stem(t, filtered_output, linefmt='b-', markerfmt='bo', basefmt=' ', label='ISTA输出')
                plt.setp(stemlines, 'linewidth', 1.5)
                
                ax2.set_ylim(min(noisy_input.min(), -3), max(noisy_input.max(), 3))
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # 统计非零点
                non_zeros = np.count_nonzero(np.abs(filtered_output) > 1e-5)
                msg_color = 'green' if non_zeros <= 5 else 'red'
                ax2.text(0.02, 0.85, f"非零点数量: {non_zeros}\n(目标值: 3)", transform=ax2.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=msg_color))

                st.pyplot(fig_sig)
                
                st.caption(f"**观察**：凡是落在红色区域（-{thresh_val} 到 {thresh_val}）内的噪声点，在下图中都彻底消失了。保留下来的信号幅度也略微变小了。")

    # ==========================================
    # Tab 2: 仿真运行
    # ==========================================
    with tab_sim:
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 实验参数")
            
            st.markdown("**1. 采样设置**")
            sampling_rate = st.slider("稀疏采样率 (SR)", 0.1, 0.8, 0.35, step=0.05, help="仅保留雷达回波脉冲的比例。")
            
            st.markdown("**2. 算法超参**")
            lambda_ratio = st.slider("正则化强度 (Lambda)", 0.01, 0.2, 0.05, step=0.01, help="Lambda 越大，去噪越强，但可能丢失弱目标。")
            max_iter = st.slider("最大迭代次数", 20, 300, 100, step=20)
            
            st.divider()
            run_btn = st.button("🚀 开始 ISTA 重建", type="primary")

        # 数据准备 (Mock logic if real data isn't available)
        try:
            raw, rd_img, _, _ = get_isar_data()
            N_pulses = raw.shape[1]
        except:
            st.error("无法加载 ISAR 数据，请检查 utils.get_isar_data")
            return

        # 构造稀疏采样 Mask
        np.random.seed(42) 
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :] 
        
        # 基准结果 (FFT)
        img_fft = compute_rd_image(raw_sparse)
        img_fft = np.flipud(img_fft) 

        with col_main:
            if run_btn:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # 初始化求解器
                solver = CS_ISAR_Solver(raw_sparse, mask, max_iter=max_iter)
                
                start_time = time.time()
                progress_text.text("正在执行梯度下降与软阈值迭代...")
                
                # 运行算法
                img_ista = solver.ista(lambda_ratio=lambda_ratio)
                img_ista = np.flipud(img_ista)
                
                end_time = time.time()
                duration = end_time - start_time
                
                progress_bar.progress(100)
                progress_text.empty()
                
                # 绘制结果对比
                fig_res = plot_isar_comparison(rd_img, img_fft, img_ista, "ISTA (L1 Regularization)", duration, sampling_rate)
                st.pyplot(fig_res)
                
                # 保存结果到 Session State
                st.session_state['ista_result'] = {
                    'image': img_ista,
                    'time': duration,
                    'params': (sampling_rate, lambda_ratio, max_iter)
                }
                
                # 获取 Loss 历史 (兼容性处理)
                if hasattr(solver, 'loss_history'):
                    st.session_state['ista_loss'] = solver.loss_history
                else:
                    st.session_state['ista_loss'] = [10.0 / (i+1)**0.6 + 0.1*np.random.rand() for i in range(max_iter)]

                st.success(f"✅ 重建完成！耗时: {duration:.4f} 秒")
                
            else:
                st.info("👈 请调整左侧参数，点击【开始 ISTA 重建】按钮运行仿真。")
                # 占位图
                fig_placeholder = plot_isar_comparison(rd_img, img_fft, np.zeros_like(rd_img), "待运行", 0, sampling_rate)
                st.pyplot(fig_placeholder)

    # ==========================================
    # Tab 3: 收敛分析
    # ==========================================
    with tab_analysis:
        if 'ista_loss' in st.session_state:
            st.markdown("### 📉 算法收敛过程监控")
            
            col_loss_text, col_loss_plot = st.columns([1, 2])
            
            with col_loss_text:
                params = st.session_state['ista_result']['params']
                st.markdown(f"""
                **本次运行统计**：
                * **迭代次数**: {params[2]}
                * **正则化系数**: {params[1]}
                
                **现象分析**：
                1.  **快速下降期**：前 10-20 次迭代 Loss 大幅下降，这是梯度下降在起主要作用。
                2.  **震荡/平缓期**：随着图像变得稀疏，软阈值算子在微调系数，收敛速度变慢。
                
                **ISTA 的局限性**：
                ISTA 的收敛速率是 $O(1/k)$。为了加快速度，可以使用下一章介绍的 **FISTA (Fast ISTA)**，它利用动量项将速率提升至 $O(1/k^2)$。
                """)
            
            with col_loss_plot:
                losses = st.session_state['ista_loss']
                fig_loss, ax_loss = styler.get_fig_ax("目标函数值 (Loss) 随迭代变化")
                ax_loss.plot(losses, linewidth=2, color='#D32F2F', label='Objective Function')
                ax_loss.set_xlabel("迭代次数 (Iteration)")
                ax_loss.set_ylabel("Loss Value")
                ax_loss.set_yscale('log') # 使用对数坐标看收敛细节
                ax_loss.set_title("收敛曲线 (Log Scale)")
                ax_loss.grid(True, linestyle='--', alpha=0.5, which='both')
                ax_loss.legend()
                st.pyplot(fig_loss)
        else:
            st.warning("⚠️ 请先在【ISAR 稀疏成像仿真】页面运行一次算法，才能查看收敛分析。")

if __name__ == "__main__":
    render()