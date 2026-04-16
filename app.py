import streamlit as st
import pandas as pd
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank


st.set_page_config(page_title="KM Survival AI", layout="centered")

st.title("📊 Survival Analysis AI Extraction")
st.markdown("""
Upload a Kaplan–Meier curve, and the AI will automatically extract the data 
and compute the **p-value** for the log-rank test.
""")


uploaded_file = st.file_uploader("Upload KM plot (JPG/PNG)", type=["png", "jpg", "jpeg"])


@st.cache_data(show_spinner=False)
def get_ai_data(file_content):
    return extract_data_from_km_image(file_content)

if uploaded_file:
    # Preview Image
    st.image(uploaded_file, caption="Uploaded Kaplan-Meier Curve", use_container_width=True)
    
    # Click the bottom
    
    if st.button("🚀 Start AI Analysis", type="primary"):
        with st.spinner("AI is analyzing the curve... please wait."):
            
            data = get_ai_data(uploaded_file)
        
        if "error" in data:
            st.error(f"Extraction failed: {data['error']}")
            st.warning("Hint: If it's a 429 error, please wait 1-2 minutes.")
        else:
            st.success("Data extraction successful！")
            
            # Display data
            with st.expander("View Raw Extracted JSON Data"):
                st.json(data)
            
            # Statistical Computing
            st.subheader("📈 Statistical Analysis Result")
            try:
                p_value = calculate_log_rank(data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Log-rank P-value", value=f"{p_value:.4f}")
                
                with col2:
                    if p_value < 0.05:
                        st.write("✅ **Statistically Significant** (p < 0.05)")
                    else:
                        st.write("❌ **Not Statistically Significant** (p >= 0.05)")
                
                st.info("The p-value is calculated from the reconstructed data.")
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")

else:
    st.info("Please upload an image and then click the analysis button.")

st.markdown("---")
st.caption("Developed for Biostatistics & Data Science Project | Weill Cornell Medicine")
