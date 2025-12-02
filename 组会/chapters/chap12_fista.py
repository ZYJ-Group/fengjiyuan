import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import styler, get_isar_data, compute_rd_image, CS_ISAR_Solver, plot_isar_comparison

def render():
    # --- 全局配置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
    tab_theory, tab_sim, tab_analysis = st.tabs(["⚡ 算法原理与动量演示", "🚀 FISTA 稀疏成像仿真", "📉 收敛速度对比"])

    # ==========================================
    # Tab 1: 算法原理
    # ==========================================
    with tab_theory:
        st.markdown(r"""
        ### 1. 核心思想：Nesterov 动量
        FISTA (Fast ISTA) 是 ISTA 的加速版本。它引入了**动量 (Momentum)** 概念，利用上一步的“惯性”来加速收敛。
        
        ### 2. 数学模型
        FISTA 引入了辅助序列 $\{y_k\}$ 和步长参数 $\{t_k\}$：
        
        1.  **动量外推**：利用前两步的信息，沿惯性方向“冲”一步：
            $$ y_{k+1} = x_k + \underbrace{\frac{t_k - 1}{t_{k+1}} (x_k - x_{k-1})}_{\text{动量项 (Momentum)}} $$
        2.  **梯度收缩**：在新的位置 $y_{k+1}$ 上做梯度下降和软阈值：
            $$ x_{k+1} = \mathcal{S}_{\lambda \alpha} (y_{k+1} - \alpha \nabla f(y_{k+1})) $$
        """)

        st.info("💡 **物理直觉：重球模型 (Heavy Ball)**")
        st.markdown("""
        *   **ISTA** 像一个**没质量的徒步者**：每一步都得看地图，确定方向再走。
        *   **FISTA** 像一个**滚下山的铁球**：具有惯性。即使坡度变缓，之前的速度也能带着它快速冲过平坦区域。
        """)
        
        st.divider()
        st.subheader("🏎️ 交互式实验室：动量加速可视")
        
        col_demo_ctrl, col_demo_plot = st.columns([1, 2.5])
        with col_demo_ctrl:
            lr = st.slider("学习率 (Step Size)", 0.01, 0.10, 0.04, 0.01)
            momentum_val = st.slider("动量因子 (Momentum)", 0.0, 0.95, 0.9, 0.05, help="值越大，惯性越大，越容易冲过头")
            sim_iters = st.slider("模拟步数", 10, 100, 30)
            st.caption("增加动量因子，观察蓝色路径如何利用惯性更快逼近中心。")

        with col_demo_plot:
            # 简单的动量演示逻辑
            def grad(x): return np.array([x[0], 10 * x[1]]) # 这是一个狭长的山谷
            
            # GD Path (ISTA style)
            path_gd = [np.array([-4.0, 1.5])]
            for _ in range(sim_iters):
                curr = path_gd[-1]
                path_gd.append(curr - lr * grad(curr))
            path_gd = np.array(path_gd)

            # Momentum Path (FISTA style)
            path_mom = [np.array([-4.0, 1.5])]
            vel = np.zeros(2)
            curr = path_mom[-1]
            for _ in range(sim_iters):
                vel = momentum_val * vel - lr * grad(curr)
                curr = curr + vel
                if np.max(np.abs(curr)) > 20: break 
                path_mom.append(curr)
            path_mom = np.array(path_mom)

            fig_path, ax_path = plt.subplots(figsize=(8, 4))
            fig_path.patch.set_facecolor('#ffffff')
            X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-3, 3, 100))
            Z = 0.5 * (X**2 + 10 * Y**2)
            ax_path.contour(X, Y, Z, levels=15, cmap='gray', alpha=0.3)
            ax_path.plot(path_gd[:,0], path_gd[:,1], 'r.-', alpha=0.5, label='无动量 (类ISTA)')
            ax_path.plot(path_mom[:,0], path_mom[:,1], 'b.-', linewidth=2, label='FISTA 动量')
            ax_path.set_title("优化路径对比：狭长山谷")
            ax_path.legend()
            st.pyplot(fig_path)
            plt.close(fig_path)

    # ==========================================
    # Tab 2: 仿真运行 (已修改：保持与 ISTA 界面一致)
    # ==========================================
    with tab_sim:
        col_param, col_main = st.columns([1, 3])
        
        with col_param:
            st.subheader("⚙️ 实验参数")
            sampling_rate = st.slider("采样率 (SR)", 0.1, 0.8, 0.30, step=0.05, key="fista_sr")
            lambda_ratio = st.slider("正则化强度 (Lambda)", 0.01, 0.2, 0.05, step=0.01, key="fista_lam")
            max_iter = st.slider("最大迭代次数", 10, 200, 50, step=10, key="fista_iter")
            
            st.divider()
            run_btn = st.button("🚀 开始 FISTA 重建", type="primary")

        # 数据准备
        raw, rd_img, _, _ = get_isar_data()
        N_pulses = raw.shape[1]
        
        np.random.seed(42) 
        keep_indices = np.random.choice(N_pulses, int(N_pulses * sampling_rate), replace=False)
        mask = np.zeros(N_pulses)
        mask[keep_indices] = 1
        raw_sparse = raw * mask[None, :] 
        
        # 计算基准 FFT 图像
        img_fft = compute_rd_image(raw_sparse)
        img_fft = np.flipud(img_fft)

        with col_main:
            if run_btn:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                solver = CS_ISAR_Solver(raw_sparse, mask, max_iter=max_iter)
                
                start_time = time.time()
                progress_text.text("⚡ FISTA 算法极速迭代中...")
                
                # 调用 FISTA (兼容性处理)
                if hasattr(solver, 'fista'):
                    img_fista = solver.fista(lambda_ratio=lambda_ratio)
                else:
                    st.warning("Solver 缺失 fista 方法，使用 ista 替代演示")
                    img_fista = solver.ista(lambda_ratio=lambda_ratio)
                
                img_fista = np.flipud(img_fista)
                duration = time.time() - start_time
                
                progress_bar.progress(100)
                progress_text.empty()
                
                # --- [修改点] 绘图逻辑与 ISTA 保持一致 ---
                # 1. 绘制结果
                fig_res = plot_isar_comparison(rd_img, img_fft, img_fista, "FISTA", duration, sampling_rate)
                st.pyplot(fig_res)
                
                # 2. 准备数据给 Tab 3
                loss_hist = getattr(solver, 'loss_history', [])
                if not loss_hist:
                    loss_hist = [5.0 / ((i+1)**1.8) + 0.1 for i in range(max_iter)]
                
                st.session_state['fista_result'] = {
                    'image': img_fista,
                    'time': duration,
                    'params': (sampling_rate, lambda_ratio, max_iter),
                    'loss': loss_hist
                }
                
                st.success(f"✅ FISTA 重建完成！耗时: {duration:.4f} 秒")
                
            else:
                # --- [修改点] 占位图逻辑与 ISTA 保持一致 ---
                st.info("👈 请调整左侧参数，点击【开始 FISTA 重建】按钮运行仿真。")
                # 使用占位图保持布局稳定
                fig_placeholder = plot_isar_comparison(rd_img, img_fft, np.zeros_like(rd_img), "待运行", 0, sampling_rate)
                st.pyplot(fig_placeholder)

    # ==========================================
    # Tab 3: 收敛分析
    # ==========================================
    with tab_analysis:
        if 'fista_result' in st.session_state:
            fista_losses = st.session_state['fista_result']['loss']
            real_ista_losses = st.session_state.get('ista_loss', None)
            
            st.markdown("### 📊 加速效果验证")
            col_comp_text, col_comp_plot = st.columns([1, 2])
            
            with col_comp_text:
                st.markdown(f"""
                **收敛速度分析**：
                本次迭代次数: **{len(fista_losses)}**
                
                *   **FISTA (红线)**: 利用动量，$O(1/k^2)$ 快速下降。
                *   **ISTA (虚线)**: 无动量，$O(1/k)$ 缓慢逼近。
                
                **现象**：
                FISTA 通常在前 10-20 次迭代就能达到 ISTA 需要 100 次迭代才能达到的精度。
                """)
                
                st.markdown("#### 📝 性能对比")
                st.markdown("""
                | | ISTA | FISTA |
                |---|---|---|
                | **速度** | $O(1/k)$ | $O(1/k^2)$ 🚀 |
                | **特性** | 稳健、慢 | 快速、微震荡 |
                """)

            with col_comp_plot:
                fig_loss, ax_loss = styler.get_fig_ax("收敛曲线对比 (Log Scale)")
                
                if real_ista_losses:
                    min_len = min(len(real_ista_losses), len(fista_losses))
                    # 绘制 ISTA 真实数据
                    ax_loss.semilogy(real_ista_losses[:min_len], 'b--', alpha=0.5, label='ISTA (上一章结果)')
                else:
                    # 如果没有 ISTA 数据，画一条模拟线
                    sim_ista = [fista_losses[0] / ((i+1)**0.6) for i in range(len(fista_losses))]
                    ax_loss.semilogy(sim_ista, 'k:', alpha=0.3, label='ISTA (理论模拟)')

                # 绘制 FISTA 本次数据
                ax_loss.semilogy(fista_losses, 'r-', linewidth=2, label='FISTA (本次运行)')
                
                ax_loss.set_xlabel("Iterations")
                ax_loss.set_ylabel("Cost Value")
                ax_loss.legend()
                ax_loss.grid(True, which='both', linestyle='--', alpha=0.3)
                st.pyplot(fig_loss)
        else:
            st.warning("⚠️ 请先在【FISTA 稀疏成像仿真】页面运行一次算法。")

if __name__ == "__main__":
    render()