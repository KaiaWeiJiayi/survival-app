import streamlit as st
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank

# Set up the Streamlit page configuration
st.set_page_config(page_title="KM Data Extractor & Log-rank", layout="wide")

st.title("Survival Analysis AI Extraction and Analysis Tool")
st.markdown("Upload a Kaplan–Meier curve, and the AI will automatically extract the data and compute the p-value for the log-rank test.")

# File uploader for the user to provide the KM curve image
uploaded_file = st.file_uploader("请上传 KM 曲线图 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image in the left column
        st.image(uploaded_file, caption="用户上传的原始图片", use_container_width=True)

    with col2:
        if st.button("🚀 开始解析图片与计算", type="primary"):
            with st.spinner("AI 正在努力读取坐标系和图例，请稍候..."):
                # 1. Extract data using the LLM
                extracted_json = extract_data_from_km_image(uploaded_file)

                if "error" in extracted_json:
                    st.error(f"提取失败: {extracted_json['error']}")
                else:
                    st.success("数据提取成功！")
                    st.json(extracted_json)

                    st.subheader("统计分析结果")
                    try:
                        # 2. Calculate the statistical results (P-value)
                        p_value = calculate_log_rank(extracted_json)
                        st.metric(label="Log-rank Test P-value", value=f"{p_value:.5f}")

                        if p_value < 0.05:
                            st.info("💡 结论：两组之间存在显著的统计学差异 (P < 0.05)。")
                        else:
                            st.info("💡 结论：两组之间未见显著的统计学差异 (P >= 0.05)。")
                    except Exception as e:
                        st.warning("提取的数据格式不足以计算 P 值，请检查 AI 提取结果的准确性。")