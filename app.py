import streamlit as st
import pandas as pd
from llm_extractor import extract_data_from_km_image
from stats_calculator import calculate_log_rank, calculate_bucher_method
from stats_calculator import plot_reconstructed_km

st.set_page_config(page_title="KM Survival AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Cornell Red */
    div.stButton > button:first-child {
        background-color: #b31b1b;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #8a1515;
        color: white;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 2px solid #b31b1b;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.image("wcm_logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Analysis Modules")

selected_module = st.sidebar.radio(
    "Select an analysis:",
    ["📌 Single Trial Analysis", "🌟 Indirect Comparison (Bucher)"]
)

st.markdown(
    """
    <h1 style='text-align: center; color: #b31b1b;'>
        📊 AI-Powered Survival Analysis System
    </h1>
    <p style='text-align: center; font-size: 1.2em; color: #555;'>
        Advanced Kaplan-Meier Digitization & Indirect Treatment Comparison
    </p>
    <hr>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown("Department of Population Health Sciences")
st.sidebar.caption("👨‍💻 Developed by: Jiayi Wei")


# ==========================================
# Module 1
# ==========================================
if selected_module == "📌 Single Trial Analysis":
    st.header("📊 Single Trial Analysis")
    st.markdown("Upload a single KM curve to automatically extract data and compute the P-value.")
    
    uploaded_file = st.file_uploader("Upload KM plot (JPG/PNG)", type=["png", "jpg", "jpeg"], key="single_upload")

    @st.cache_data(show_spinner=False)
    def get_ai_data(file_content):
        return extract_data_from_km_image(file_content)

    if uploaded_file:
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption="Uploaded Kaplan-Meier Curve", use_container_width=True)
        
        with col_res:
            if st.button("🚀 Start AI Analysis", key="btn_single"):
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
                        p_value, df_reconstructed = calculate_log_rank(data)
                        
                        st.metric(label="Log-rank P-value", value=f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.write("✅ **Statistically Significant** (p < 0.05)")
                        else:
                            st.write("❌ **Not Statistically Significant** (p >= 0.05)")
                        
                        st.info(f"Analysis performed: {'Multivariate' if len(data) > 2 else 'Standard'} Log-rank Test with Linear Interpolation")

                        # ==========================================
                        # Validation Area
                        # ==========================================
                        st.markdown("---")
                        st.subheader("🔍 Extraction Validation")
                        st.markdown("Compare the original plot with the curves generated from the AI-extracted Pseudo-IPD.")
                        

                        val_col1, val_col2 = st.columns(2)
                        
                        with val_col1:
                            st.markdown("**Original Extracted Image**")
                            # Initial km plot
                            st.image(uploaded_file, width="stretch")
                            
                        with val_col2:
                            st.markdown("**Reconstructed Plot (lifelines)**")
                            with st.spinner("Generating validation plot..."):
                                # import df_reconstructed
                                validation_fig = plot_reconstructed_km(df_reconstructed)
                               
                                st.pyplot(validation_fig)
                        
                        st.markdown("---")
                        st.subheader("📋 Reconstructed Patient-Level Data")
                        st.dataframe(df_reconstructed, use_container_width=True)
                        
                        csv = df_reconstructed.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Data as CSV",
                            data=csv,
                            file_name='reconstructed_survival_data.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Stats Error: {str(e)}")

# ==========================================
# Module 2
# ==========================================
elif selected_module == "🌟 Indirect Comparison (Bucher)":
    st.header("🌉 Indirect Treatment Comparison")
    st.markdown("Compare **Treatment A vs Treatment C** indirectly by leveraging a common **Treatment B** from two separate trials.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trial 1 (e.g., A vs B)")
        file1 = st.file_uploader("Upload Trial 1 KM plot", type=["png", "jpg", "jpeg"], key="file1")
        if file1: st.image(file1, use_container_width=True)
        
    with col2:
        st.subheader("Trial 2 (e.g., B vs C)")
        file2 = st.file_uploader("Upload Trial 2 KM plot", type=["png", "jpg", "jpeg"], key="file2")
        if file2: st.image(file2, use_container_width=True)

    if file1 and file2:
        if st.button("🚀 Step 1: Digitize Both Trials", key="btn_double"):
            with st.spinner("AI is analyzing both curves... this might take a moment."):
                st.session_state['data1'] = extract_data_from_km_image(file1)
                st.session_state['data2'] = extract_data_from_km_image(file2)
        
        if 'data1' in st.session_state and 'data2' in st.session_state:
            d1, d2 = st.session_state['data1'], st.session_state['data2']
            
            if "error" in d1 or "error" in d2:
                st.error("Extraction failed for one or both images. Please try again.")
            else:
                st.success("✅ Both trials digitized successfully!")
                st.markdown("#### 🎯 Step 2: Define the Comparison Network")
                
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

                if st.button("🔬 Step 3: Run Bucher Method (A vs C)"):
                    with st.spinner("Fitting Cox Proportional Hazards models..."):
                        try:
                            results = calculate_bucher_method(
                                d1, treat_a, treat_b1, 
                                d2, treat_c, treat_b2
                            )
                            
                            st.success("Indirect Comparison Complete!")
                            res_ac = results["Indirect (A vs C)"]
                            
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Indirect Hazard Ratio (A vs C)", f"{res_ac['HR']:.2f}")
                            col_m2.metric("95% Confidence Interval", f"[{res_ac['CI_Lower']:.2f}, {res_ac['CI_Upper']:.2f}]")
                            col_m3.metric("P-value", f"{res_ac['P_Value']:.4f}")
                            
                            if res_ac['P_Value'] < 0.05:
                                st.info(f"**Conclusion:** There is a statistically significant difference between {treat_a} and {treat_c}.")
                            else:
                                st.warning(f"**Conclusion:** No statistically significant difference between {treat_a} and {treat_c}.")
                                
                            with st.expander("View Detailed Math breakdown"):
                                math_results = {k: v for k, v in results.items() if k != "DataFrames"}
                                st.json(math_results)
                            
                            # ==========================================
                            # Validation Area
                            # ==========================================
                            st.markdown("---")
                            st.subheader("🔍 Trials Data Validation")
                            st.markdown("Verify the reconstruction accuracy for both input trials before comparison.")
                            
                            df1 = results["DataFrames"]["Trial 1"]
                            df2 = results["DataFrames"]["Trial 2"]
                            
                            # Container
                            t1_container = st.container(border=True)
                            t2_container = st.container(border=True)
                            
                            with t1_container:
                                st.markdown(f"### 📄 Trial 1: {treat_a} vs {treat_b1}")
                                v1_col_img, v1_col_plot = st.columns(2)
                                with v1_col_img:
                                    st.markdown("Original Image")
                                    st.image(file1, use_container_width=True)
                                with v1_col_plot:
                                    st.markdown("Reconstructed Plot")
                                    fig1 = plot_reconstructed_km(df1)
                                    st.pyplot(fig1)
                                    
                                st.dataframe(df1, use_container_width=True, height=200)
                                st.download_button("📥 Download Trial 1 CSV", data=df1.to_csv(index=False).encode('utf-8'), file_name='trial1_data.csv', mime='text/csv')

                            st.markdown("---")

                            with t2_container:
                                st.markdown(f"### 📄 Trial 2: {treat_c} vs {treat_b2}")
                                v2_col_img, v2_col_plot = st.columns(2)
                                with v2_col_img:
                                    st.markdown("Original Image")
                                    st.image(file2, use_container_width=True)
                                with v2_col_plot:
                                    st.markdown("Reconstructed Plot")
                                    fig2 = plot_reconstructed_km(df2)
                                    st.pyplot(fig2)
                                    
                                st.dataframe(df2, use_container_width=True, height=200)
                                st.download_button("📥 Download Trial 2 CSV", data=df2.to_csv(index=False).encode('utf-8'), file_name='trial2_data.csv', mime='text/csv')

                        # except
                        except Exception as e:
                            st.error(f"Statistical computation failed: {str(e)}")
