from google import genai
import json
from PIL import Image
import streamlit as st

# Securely fetch API Key
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

def extract_data_from_km_image(image_file):
    """
    Extract survival data from a KM curve using the new Gemini SDK.
    """
    prompt = """
    You are a biostatistician. Analyze this Kaplan-Meier curve. 
    Extract the time points, survival probabilities, and numbers at risk for each group.
    Output strictly in JSON format as follows:
    {
      "Group_A": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}],
      "Group_B": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}]
    }
    """
    
    try:
        img = Image.open(image_file)
        
        # We use a very direct call to avoid 404 path issues
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, img]
        )
        
        text = response.text.strip()
        # Extract JSON if LLM wraps it in markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        return json.loads(text)
        
    except Exception as e:
        # If 1.5-flash fails, it might be a regional model naming issue
        return {"error": str(e)}
