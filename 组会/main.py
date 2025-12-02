import streamlit as st
from chapters import (
    chap01_norms, chap02_convex, chap03_funcs, 
    chap04_conj, chap05_subgrad, chap06_theory, chap07_algos, 
    chap08_cs, chap09_isar, chap10_lasso, 
    chap11_ista, chap12_fista, chap13_admm, chap14_ctn
)

# 设置页面配置 (必须是第一个 Streamlit 命令)
st.set_page_config(
    page_title="最优化与压缩感知笔记",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("📡 导航菜单")
    
    # 定义章节结构
    structure = {
        "第一部分：数学基础": {
            "1. 范数与几何 (Norms)": chap01_norms,
            "2. 凸集 (Convex Sets)": chap02_convex,
            "3. 凸函数 (Convex Functions)": chap03_funcs,
            "4. 共轭与对偶 (Conjugate & Duality)": chap04_conj,
            "5. 次梯度 (Subgradients)": chap05_subgrad,
            "6. 最优性条件 (Optimality)": chap06_theory,
        },
        "第二部分：问题建模": {
            "7. 优化算法谱系 (Overview)": chap07_algos,
            "8. 压缩感知原理 (CS)": chap08_cs,
            "9. ISAR 测量矩阵构建": chap09_isar,
            "10. LASSO 回归 (数学模型)": chap10_lasso,
        },
        "第三部分：求解算法 (ISAR应用)": {
            "11. 梯度类算法 (ISTA)": chap11_ista,
            "12. 加速梯度算法 (FISTA)": chap12_fista,
            "13. 算子分裂算法 (ADMM)": chap13_admm,
            "14. 总结": chap14_ctn,
        }
    }

    # 侧边栏渲染
    selected_section = st.sidebar.selectbox("选择模块", list(structure.keys()))
    st.sidebar.markdown("---")
    selected_chapter = st.sidebar.radio("选择章节", list(structure[selected_section].keys()))

    # 页面渲染
    st.markdown(f"## {selected_chapter}")
    st.markdown("---")
    
    # 调用对应章节的 render 函数
    module = structure[selected_section][selected_chapter]
    module.render()

    # 底部统一页脚
    st.markdown("---")

if __name__ == "__main__":
    main()