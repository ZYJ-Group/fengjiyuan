import streamlit as st

def render():
    st.markdown("""
    本章汇总了个人在 **稀疏信号处理** 与 **凸优化** 方向的核心参考文献与工具。
    
    """)

    st.divider()

    # 使用 Tabs 分类，更像一个学术导航页
    tab_math, tab_course, tab_tools = st.tabs(["📖 数学与理论基础", "📺 视频课程推荐", "🛠️ 工程实现工具箱"])

    # ==========================================
    # Tab 1: 理论基础 (包含指定的 PKU 链接)
    # ==========================================
    with tab_math:
        st.subheader("国内优质教材(中文)")
        
        # 重点推荐部分 - 根据要求添加的链接
        st.success("""
        **🔥 组内强推：最优化理论与算法 (北大文再文教授)**
        
        这是北京大学文再文老师组编写的《最优化：建模、算法与理论》。相比于国外的教材，这本在线讲义对 **ADMM**、**算子分裂** 以及 **流形优化** 的讲解非常前沿且细致，非常适合做案头参考。
        
        👉 **在线阅读地址**: [http://faculty.bicmr.pku.edu.cn/~wenzw/optbook...](http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/contents/contents.html#1)
        """)

        st.markdown("---")
        st.subheader("经典英文原著")
        
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown("#### Convex Optimization")
            st.caption("Boyd & Vandenberghe")
            st.markdown("""
            * **定位**：优化领域的《圣经》。
            * **必读章节**：Ch.2 (凸集), Ch.3 (凸函数), Ch.5 (对偶理论)。
            * **备注**：做理论推导遇到瓶颈时，翻它准没错。
            """)
            st.link_button("PDF 下载 (Stanford)", "https://web.stanford.edu/~boyd/cvxbook/")

        with col_b2:
            st.markdown("#### Proximal Algorithms")
            st.caption("Neal Parikh & Stephen Boyd")
            st.markdown("""
            * **定位**：针对非平滑优化（L1正则）的手册。
            * **必读理由**：如果不理解 **Proximal Operator (近端算子)**，就无法真正看懂 ISTA 和 ADMM 的收敛证明。
            * **适用**：写论文需要推导算法收敛性时。
            """)
            st.link_button("PDF 下载", "https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf")

    # ==========================================
    # Tab 2: 视频课程
    # ==========================================
    with tab_course:
        st.info("💡 **建议**：吃饭或跑代码间隙观看，建立物理直觉，不要纠结于每一个公式细节。")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            with st.container(border=True):
                st.markdown("#### 1. Steve Brunton (YouTube)")
                st.caption("University of Washington")
                st.markdown("""
                * **关键词**: SVD, Compressed Sensing, Data-Driven Science.
                * **风格**: 板书极美，结合 Python 代码，非常适合建立直观的**物理图像**。
                * **适合阶段**: 入门 / 寻找灵感。
                """)
                st.link_button("👉 前往频道", "https://www.youtube.com/@Eigensteve")

        with col_v2:
            with st.container(border=True):
                st.markdown("#### 2. EE364a: Convex Optimization")
                st.caption("Stephen Boyd (Stanford)")
                st.markdown("""
                * **关键词**: 凸优化, 对偶理论, KKT条件.
                * **风格**: 硬核数学，语速适中，逻辑闭环。
                * **适合阶段**: 研一上学期系统学习。
                """)
                st.link_button("👉 课程主页", "https://web.stanford.edu/class/ee364a/")

    # ==========================================
    # Tab 3: 工程工具
    # ==========================================
    with tab_tools:
        st.markdown("### 🚫 不要重复造轮子")
        st.markdown("在验证想法（Idea Verification）阶段，请优先使用成熟的库，确定算法有效后再尝试手写 C++/CUDA 加速。")
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("CVXPY", "Ground Truth", "基准验证")
            st.markdown("""
            **用途**：
            当你不确定自己写的 ADMM 是否收敛到全局最优时，用 CVXPY 跑一遍作为标准答案对比。
            """)
            
        with c2:
            st.metric("PyLops", "Linear Ops", "大尺度运算")
            st.markdown("""
            **用途**：
            雷达/地震数据太大，无法构建显式矩阵 $A$。PyLops 提供了 `A * x` 和 `A.H * y` 的算子接口。
            """)
            
        with c3:
            st.metric("Deep Unfolding", "PyTorch", "论文热点")
            st.markdown("""
            **用途**：
            将 ISTA/ADMM 展开成神经网络 (LISTA, ADMM-Net)。这是目前发这方面 Paper 的主流方向之一。
            """)

        with c4:
            st.metric("Matplotlib", "Visualization", "绘图排版")
            st.markdown("""
            **用途**：
            Paper 里的图必须矢量化且美观。建议学习 `plt.rcParams` 配置，统一全组的绘图风格。
            """)

    st.divider()

if __name__ == "__main__":
    render()