import streamlit as st
import pandas as pd
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank

# 1. Page Configuration
st.set_page_config(page_title="KM Survival AI", layout="centered")

st.title("📊 Survival Analysis AI Extraction")
st.markdown("""
Upload a Kaplan–Meier curve, and the AI will automatically extract the data 
and compute the **p-value** for the log-rank test.
""")

# 2. Sidebar / Settings
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Upload a clear KM plot image.
2. The AI (Gemini 2.0 Lite) will extract data points.
3. Log-rank p-value will be calculated automatically.
""")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload KM plot (JPG/PNG)", type=["png", "jpg", "jpeg"])


@st.cache_data(show_spinner="AI is analyzing the curve... please wait.")
def get_ai_data(file_content):
    # Return function in llm_extractor.py 
    return extract_data_from_km_image(file_content)

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Kaplan-Meier Curve", use_container_width=True)
    
    # Trigger AI Extraction
    with st.spinner("Processing with Gemini 2.0 Lite..."):
        
        data = get_ai_data(uploaded_file)
    
    if "error" in data:
        st.error(f"Extraction failed: {data['error']}")
        st.warning("Hint: If it's a 429 error, please wait 1-2 minutes without clicking.")
    else:
        st.success("Data extraction successful！")
        
        # 4. Display Extracted Data
        with st.expander("View Raw Extracted JSON Data"):
            st.json(data)
        
        # 5. Statistical Analysis
        st.subheader("📈 Statistical Analysis Result")
        try:
            p_value = calculate_log_rank(data)
            
            # Formatting the result
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Log-rank P-value", value=f"{p_value:.4f}")
            
            with col2:
                if p_value < 0.05:
                    st.write("✅ **Statistically Significant** (p < 0.05)")
                else:
                    st.write("❌ **Not Statistically Significant** (p >= 0.05)")
            
            st.info("The p-value is calculated based on the reconstructed patient-level data from the extracted curves.")
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.write("The extracted data format might be insufficient to compute the p-value. Please check the JSON above.")

else:
    st.info("Please upload an image to start the analysis.")

# Footer
st.markdown("---")
st.caption("Developed for Biostatistics & Data Science Project | Weill Cornell Medicine")
