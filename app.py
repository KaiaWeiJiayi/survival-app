import streamlit as st
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank

# Set up the Streamlit page configuration
st.set_page_config(page_title="KM Data Extractor & Log-rank", layout="wide")

st.title("Survival Analysis AI Extraction and Analysis Tool")
st.markdown("Upload a Kaplan–Meier curve, and the AI will automatically extract the data and compute the p-value for the log-rank test.")

# File uploader for the user to provide the KM curve image
uploaded_file = st.file_uploader("Upload KM plot (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image in the left column
        st.image(uploaded_file, caption="User-uploaded original image", use_container_width=True)

    with col2:
        if st.button("Start analyzing the image and performing calculations", type="primary"):
            with st.spinner("The AI is working to read the coordinate system and legend. Please wait..."):
                # 1. Extract data using the LLM
                extracted_json = extract_data_from_km_image(uploaded_file)

                if "error" in extracted_json:
                    st.error(f"Extraction failed: {extracted_json['error']}")
                else:
                    st.success("Data extraction successful！")
                    st.json(extracted_json)

                    st.subheader("Statistical analysis result")
                    try:
                        # 2. Calculate the statistical results (P-value)
                        p_value = calculate_log_rank(extracted_json)
                        st.metric(label="Log-rank Test P-value", value=f"{p_value:.5f}")

                        if p_value < 0.05:
                            st.info("There is a statistically significant difference between the two groups. (P < 0.05)")
                        else:
                            st.info("No statistically significant difference was observed between the two groups. (P >= 0.05)")
                    except Exception as e:
                        st.warning("The extracted data format is insufficient to compute the p-value. Please check the accuracy of the AI-extracted results.")
