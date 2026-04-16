import streamlit as st
import pandas as pd
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank
# from stats_calculator import calculate_bucher_method (等我们写完后端再取消注释)

st.set_page_config(page_title="KM Survival AI", layout="wide")

st.title("📊 Survival Analysis AI & Indirect Comparison")
st.markdown("Digitize Kaplan-Meier curves using AI and perform advanced survival statistics.")

# 创建两个平行的标签页
tab1, tab2 = st.tabs(["📌 Single Trial Analysis", "🌟 Extra Credit: Indirect Comparison (Bucher)"])

# ==========================================
# TAB 1: 你原有的核心功能 (完全保留)
# ==========================================
with tab1:
    st.markdown("Upload a single KM curve to automatically extract data and compute the P-value.")
    uploaded_file = st.file_uploader("Upload KM plot (JPG/PNG)", type=["png", "jpg", "jpeg"], key="single_upload")

    @st.cache_data(show_spinner=False)
    def get_ai_data(file_content):
        return extract_data_from_km_image(file_content)

    if uploaded_file:
        col_img, col_res = st.columns([1, 1])
        with col_img:
            # 修复了弃用警告，使用 width="stretch"
            st.image(uploaded_file, caption="Uploaded Kaplan-Meier Curve", width="stretch")
        
        with col_res:
            if st.button("🚀 Start AI Analysis", type="primary", key="btn_single"):
                with st.spinner("AI is digitizing the curve..."):
                    data = get_ai_data(uploaded_file)
                
                if "error" in data:
                    st.error(f"Extraction failed: {data['error']}")
                else:
                    st.success("Data extraction successful!")
                    with st.expander("View Raw JSON"):
                        st.json(data)
                    
                    st.subheader("📈 Statistical Analysis Result")
                    try:
                        p_value = calculate_log_rank(data)
                        st.metric(label="Log-rank P-value", value=f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}")
                        if p_value < 0.05:
                            st.write("✅ **Statistically Significant** (p < 0.05)")
                        else:
                            st.write("❌ **Not Statistically Significant** (p >= 0.05)")
                    except Exception as e:
                        st.error(f"Stats Error: {str(e)}")


# ==========================================
# TAB 2: Extra Credit (Bucher 间接比较前端)
# ==========================================
with tab2:
    st.markdown("### 🌉 Indirect Treatment Comparison")
    st.markdown("Compare **Treatment A vs Treatment C** indirectly by leveraging a common **Treatment B** from two separate trials.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trial 1 (e.g., A vs B)")
        file1 = st.file_uploader("Upload Trial 1 KM plot", type=["png", "jpg", "jpeg"], key="file1")
        if file1: st.image(file1, width="stretch")
        
    with col2:
        st.subheader("Trial 2 (e.g., B vs C)")
        file2 = st.file_uploader("Upload Trial 2 KM plot", type=["png", "jpg", "jpeg"], key="file2")
        if file2: st.image(file2, width="stretch")

    # 步骤 1：一键同时提取两张图的数据
    if file1 and file2:
        if st.button("🚀 Step 1: Digitze Both Trials", type="primary", key="btn_double"):
            with st.spinner("AI is analyzing both curves... this might take a moment."):
                # 使用 session_state 保存数据，防止页面刷新丢失
                st.session_state['data1'] = extract_data_from_km_image(file1)
                st.session_state['data2'] = extract_data_from_km_image(file2)
        
        # 步骤 2：如果提取成功，展示选择网络
        if 'data1' in st.session_state and 'data2' in st.session_state:
            d1, d2 = st.session_state['data1'], st.session_state['data2']
            
            if "error" in d1 or "error" in d2:
                st.error("Extraction failed for one or both images. Please try again.")
            else:
                st.success("✅ Both trials digitized successfully!")
                st.markdown("#### 🎯 Step 2: Define the Comparison Network")
                st.caption("Tell the system which groups act as the common comparator (Treatment B).")
                
                groups1 = list(d1.keys())
                groups2 = list(d2.keys())
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    treat_a = st.selectbox("Target: Treatment A (Trial 1)", groups1)
                with col_b:
                    st.markdown("**Bridge: Treatment B (Common)**")
                    treat_b1 = st.selectbox("Treatment B in Trial 1", groups1, index=len(groups1)-1 if len(groups1)>1 else 0)
                    treat_b2 = st.selectbox("Treatment B in Trial 2", groups2, index=0)
                with col_c:
                    treat_c = st.selectbox("Target: Treatment C (Trial 2)", groups2, index=len(groups2)-1 if len(groups2)>1 else 0)

                # 步骤 3：执行计算
                if st.button("🔬 Step 3: Run Bucher Method (A vs C)", type="primary"):
                    st.info("The UI is ready! Here we will display the Hazard Ratio (HR), 95% CI, and P-value.")
                    # 接下来我们将在 stats_calculator.py 中编写 calculate_bucher 并在此时调用它
