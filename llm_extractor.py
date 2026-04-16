from google import genai
from PIL import Image
import json
import streamlit as st

# Securely fetch the API Key from Streamlit Secrets
# Make sure you have GEMINI_API_KEY defined in your Streamlit Cloud dashboard
API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize the official Google GenAI client
client = genai.Client(api_key=API_KEY)

def extract_data_from_km_image(image_file):
    """
    Extract survival data from a Kaplan-Meier curve image using Gemini 1.5 Flash.
    """
    
    # Precise prompt to guide the multimodal LLM
    prompt = """
    You are a professional biostatistician. Please analyze this Kaplan-Meier survival curve image.
    
    Task:
    1. Identify the treatment groups (strata).
    2. Extract data points: Time (X-axis), Survival Probability (Y-axis).
    3. Use the "Numbers at Risk" table at the bottom to determine the 'at_risk' count for each time point.
    
    Output Requirement:
    Strictly output in valid JSON format only, with no explanatory text or markdown blocks. 
    Use the following structure:
    {
      "Group_Name_1": [
        {"time": 0, "survival_rate": 1.0, "at_risk": 500},
        {"time": 10, "survival_rate": 0.8, "at_risk": 400}
      ],
      "Group_Name_2": [
        {"time": 0, "survival_rate": 1.0, "at_risk": 500},
        {"time": 10, "survival_rate": 0.6, "at_risk": 300}
      ]
    }
    """

    try:
        # Load the image using Pillow
        img = Image.open(image_file)
        
        # Call the Gemini 1.5 Flash model
        # Using the direct string 'gemini-1.5-flash' to avoid path errors
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, img]
        )
        
        # Extract and clean the response text
        raw_text = response.text.strip()
        
        # Robust JSON cleaning: handle cases where LLM includes ```json ... ```
        clean_json_str = raw_text
        if "```json" in raw_text:
            clean_json_str = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            clean_json_str = raw_text.split("```")[1].split("```")[0].strip()
            
        # Parse the string into a Python dictionary
        data = json.loads(clean_json_str)
        return data

    except json.JSONDecodeError as je:
        return {"error": f"JSON Parsing Error: AI returned invalid format. Raw: {raw_text[:100]}..."}
    except Exception as e:
        # Catch 404, 429, or other API related errors
        return {"error": str(e)}
