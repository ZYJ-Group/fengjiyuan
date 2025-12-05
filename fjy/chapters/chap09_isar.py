import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.gridspec as gridspec
import platform
import time

# --- 1. 字体配置 (保持不变) ---
def configure_plots():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

configure_plots()

# --- 2. 静态绘图：矩阵维度概念图 (保持优化后的布局) ---
def draw_matrix_concept(M, N, K_sparsity=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    full_h = 3.0
    scale_factor = max(M/N, 0.15) 
    compressed_h = full_h * scale_factor
    start_x = 0.5
    
    y_box = {'x': start_x, 'y': (full_h - compressed_h)/2, 'w': 0.5, 'h': compressed_h, 'color': '#FF6B6B', 'label': f'观测 y\n({M}×1)'}
    eq_x = start_x + 0.9
    phi_x = eq_x + 0.4
    phi_box = {'x': phi_x, 'y': (full_h - compressed_h)/2, 'w': 3.0, 'h': compressed_h, 'color': '#88D8B0', 'label': f'传感矩阵 $\Phi$\n({M}×{N})'}
    mul_x = phi_x + 3.4
    x_pos_x = mul_x + 0.4
    x_box = {'x': x_pos_x, 'y': 0.0, 'w': 0.5, 'h': full_h, 'color': '#5DADE2', 'label': f'原始信号 x\n({N}×1)'}
    
    def draw_block(box):
        rect = Rectangle((box['x'], box['y']), box['w'], box['h'], facecolor=box['color'], edgecolor='#333333', lw=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] - 0.2, box['label'], ha='center', va='top', fontsize=11)
        if box['h'] > 0.5 and 'Phi' in box['label']:
            ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, "随机行抽取\n(Random Rows)", ha='center', va='center', color='white', fontsize=10)
    
    draw_block(y_box)
    draw_block(phi_box)
    draw_block(x_box)
    
    ax.text(eq_x, full_h/2, "=", ha='center', va='center', fontsize=24, color='#333')
    ax.text(mul_x, full_h/2, "×", ha='center', va='center', fontsize=24, color='#333')
    
    if K_sparsity:
        np.random.seed(10)
        spikes = np.random.choice(100, 4, replace=False)
        for s in spikes:
            y_pos = s / 100 * full_h
            ax.hlines(y_pos, x_box['x'], x_box['x']+x_box['w'], colors='white', linewidth=2)
        ax.text(x_box['x'] + x_box['w'] + 0.2, full_h * 0.8, "稀疏度 K\n(非零值)", ha='left', va='center', fontsize=10, color='#333')

    ax.set_xlim(0, x_pos_x + 2.0)
    ax.set_ylim(-1.0, full_h + 0.5)
    ax.axis('off')
    return fig

# --- 3. 动态绘图：构建过程动画 (新增组装逻辑) ---
def play_construction_animation(M_full, M_keep):
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- 1. 数据准备 ---
    N_grid = 40
    np.random.seed(42)
    # 生成全矩阵
    t_vec = np.arange(M_full)
    f_vec = np.arange(N_grid)
    Full_Matrix = np.cos(2 * np.pi * np.outer(t_vec, f_vec) / M_full)
    
    # 随机选择索引 (排序以便观察，或者乱序更真实，这里用乱序模拟随机采样)
    # 为了动画好看，我们模拟乱序采样，但最后堆叠在一起
    selected_indices = np.random.choice(M_full, M_keep, replace=False)
    # 实际上由于时间是线性的，我们按时间顺序排序索引会更符合物理直觉
    selected_indices = np.sort(selected_indices)
    
    # 初始化“空白”的传感矩阵 (用NaN填充，显示为空白)
    Growing_Matrix = np.zeros((M_keep, N_grid)) * np.nan
    
    total_frames = M_keep
    
    # --- 2. 动画循环 ---
    for step, real_idx in enumerate(selected_indices):
        # step: 当前是第几行 (0 ~ M_keep-1)
        # real_idx: 对应全矩阵的第几行 (0 ~ M_full-1)
        
        # 更新传感矩阵的数据
        Growing_Matrix[step, :] = Full_Matrix[real_idx, :]
        
        # --- 绘图 ---
        fig = plt.figure(figsize=(10, 7))
        # 布局：上部1行(时域)，下部1行分为左右两列(全矩阵 vs 传感矩阵)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1], hspace=0.3, wspace=0.2)
        
        ax_time = fig.add_subplot(gs[0, :])      # 顶部跨列
        ax_full = fig.add_subplot(gs[1, 0])      # 左下
        ax_sensing = fig.add_subplot(gs[1, 1])   # 右下
        
        # === 1. Top: 时域脉冲 ===
        ax_time.set_title(f"Step {step+1}/{M_keep}: 物理层脉冲发射 (Time Domain)", fontsize=10, fontweight='bold', loc='left')
        # 背景灰条
        ax_time.bar(np.arange(M_full), np.ones(M_full), color='#F0F0F0', width=0.8)
        # 已经选过的
        current_indices = selected_indices[:step+1]
        ax_time.bar(current_indices, np.ones(len(current_indices)), color='#FF6B6B', width=0.8)
        # 当前正在处理的 (高亮)
        ax_time.bar(real_idx, 1, color='#FFD700', width=0.8, edgecolor='black', linewidth=1.5, zorder=10)
        
        ax_time.set_xlim(-1, M_full)
        ax_time.set_yticks([])
        ax_time.set_xticks([])
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)
        ax_time.spines['left'].set_visible(False)
        
        # === 2. Bottom Left: 全字典提取 ===
        ax_full.set_title("源：全傅里叶字典 (Full Dictionary)", fontsize=10, color='#555')
        # 背景淡化
        ax_full.imshow(Full_Matrix, aspect='auto', cmap='Greys', vmin=-1, vmax=1, alpha=0.1)
        # 高亮当前选中的那一行
        # 创建一个只包含当前行的mask
        Highlight_Row = np.zeros_like(Full_Matrix) * np.nan
        Highlight_Row[real_idx, :] = Full_Matrix[real_idx, :]
        ax_full.imshow(Highlight_Row, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        
        # 画个框框住当前行
        rect_src = Rectangle((-0.5, real_idx - 0.5), N_grid, 1, fill=False, edgecolor='#FFD700', linewidth=2)
        ax_full.add_patch(rect_src)
        
        ax_full.set_ylabel(f"Full Index (0~{M_full})")
        ax_full.set_xlabel("N")
        
        # === 3. Bottom Right: 传感矩阵堆叠 ===
        ax_sensing.set_title(r"宿：传感矩阵构建 (Building $\Phi$)", fontsize=10, fontweight='bold', color='#2E86C1')
        # 显示当前构建进度
        # 为了让比例协调，我们固定显示范围为最终大小
        ax_sensing.imshow(Growing_Matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
        
        # 画个框框住最新增加的那一行
        rect_dst = Rectangle((-0.5, step - 0.5), N_grid, 1, fill=False, edgecolor='#FFD700', linewidth=2)
        ax_sensing.add_patch(rect_dst)
        
        ax_sensing.set_ylabel(f"Compressed Index (0~{step})")
        ax_sensing.set_xlabel("N")
        # 强制Y轴范围匹配最终大小，这样矩阵就是一点点填满的，而不是忽大忽小
        ax_sensing.set_ylim(M_keep-0.5, -0.5) 
        
        # === 4. 视觉连接线 (关键优化) ===
        # 从左图的当前行，连到右图的当前行
        con = ConnectionPatch(xyA=(N_grid, real_idx), xyB=(0, step), 
                              coordsA="data", coordsB="data",
                              axesA=ax_full, axesB=ax_sensing,
                              color="#FFD700", arrowstyle="-|>", mutation_scale=20, linewidth=2, linestyle="-")
        fig.add_artist(con)
        
        # 文字说明
        status_text.markdown(f"**正在处理：** 物理时刻 $t={real_idx}$ $\\rightarrow$ 矩阵行 $m={step}$")
        
        # 渲染
        frame_placeholder.pyplot(fig)
        plt.close(fig)
        
        progress = (step + 1) / total_frames
        progress_bar.progress(progress)
        
        # 动态调整速度：前几帧慢一点让用户看清，后面加速
        time.sleep(0.15 if step < 5 else 0.05)

    status_text.success("✅ 构建完成！矩阵已压缩。")
# --- 主程序 ---
def render():  
    with st.sidebar:
        st.header("⚙️ 参数控制")
        M_full_param = st.slider("全时间维度 (N_time)", 32, 128, 64)
        CR = st.slider("压缩率 (M/N)", 0.1, 0.8, 0.3)
        M_keep_param = int(M_full_param * CR)
        st.divider()
        st.metric("保留行数 (M)", M_keep_param, delta=f"-{M_full_param - M_keep_param} (丢弃)")

    tab1, tab2 = st.tabs(["📐 几何原理", "⚡ 动态构建"])

    with tab1:
        st.subheader("欠定方程的几何视角")
        fig_concept = draw_matrix_concept(M_keep_param, M_full_param, K_sparsity=True)
        st.pyplot(fig_concept)
        
        st.markdown(r"""
        #### 1. 正向过程
        如上图所示，雷达回波的获取过程可以写成线性方程：
        
        $$ \mathbf{y} = \mathbf{\Phi} \mathbf{x} + \mathbf{n} $$
        
        *   $\mathbf{x} \in \mathbb{C}^N$：我们要恢复的 **ISAR 图像**（未知）。
        *   $\mathbf{\Phi} \in \mathbb{C}^{M \times N}$：**传感矩阵**（已知）。
            *   它是由全傅里叶矩阵 $\mathbf{F}$ 中，根据动画里的“黄色框”随机挑出 $M$ 行组成的（即部分傅里叶变换）。
        *   $\mathbf{y} \in \mathbb{C}^M$：我们接收到的 **回波数据**（已知，红色块）。
        *   $\mathbf{n}$：环境噪声。
        
        #### 2. 逆向困境 (Ill-posed Problem)
        我们的目标是已知 $\mathbf{y}$ 和 $\mathbf{\Phi}$ 求 $\mathbf{x}$。
        但看上面的参数设置：**保留行数 $M$ 远小于 总维度 $N$**（例如 $64 < 256$）。
        
        这是一个 **“胖矩阵” (Fat Matrix)** 方程组，方程个数少于未知数个数。
        $\rightarrow$ **有无穷多个解！** 任何一个解都能完美拟合数据，我们不知道哪一个是真的。
        
        #### 3. 引入先验：稀疏性 (Sparsity)
        这时候我们需要物理知识：**ISAR 图像 $\mathbf{x}$ 是什么样的？**
        它通常是黑暗背景下只有几个亮点的飞机/舰船目标。这意味着 $\mathbf{x}$ 中绝大多数元素是 0。
        
        所以，我们要求解尽可能 **“最稀疏”** 的那个解。
        
        #### 4. 数学形式化 (LASSO)
        我们将上述两个目标结合起来：
        
        1.  **数据保真**：解必须符合观测数据 $\rightarrow \min \|\mathbf{y} - \mathbf{\Phi}\mathbf{x}\|_2^2$
        2.  **稀疏约束**：解必须尽可能稀疏 $\rightarrow \min \|\mathbf{x}\|_0$ (L0 范数太难求，用 L1 范数松弛) $\rightarrow \min \|\mathbf{x}\|_1$
        
        把它们加在一起，就得到了经典的 **LASSO (Least Absolute Shrinkage and Selection Operator)** 问题形式：
        
        $$ \hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \underbrace{\frac{1}{2} \|\mathbf{y} - \mathbf{\Phi}\mathbf{x}\|_2^2}_{\text{拟合误差 (数据一致性)}} + \underbrace{\lambda \|\mathbf{x}\|_1}_{\text{稀疏惩罚 (先验知识)}} $$
        
        这就是为什么我们在后面的章节（ISTA, ADMM）中都要解这个特定的数学式子。
        """)

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 动画演示\n点击下方按钮，观察从**时域随机采样**到**矩阵组装**的全过程。")
            if st.button("▶️ 运行构建动画", type="primary"):
                with col2:
                    play_construction_animation(M_full_param, M_keep_param)
            else:
                with col2:
                    st.info("👈 等待开始...")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render()