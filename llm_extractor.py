from google import genai
import json
from PIL import Image
import streamlit as st

# Fetch API Key securely from Streamlit secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize the new genai client
client = genai.Client(api_key=API_KEY)

def extract_data_from_km_image(image_file):
    """
    Extract survival data from a Kaplan-Meier curve image using a multimodal LLM.
    """
    prompt = """
    You are a professional biostatistician. Please analyze this Kaplan-Meier survival curve.
    I need you to extract the data points for each treatment group (e.g., Group A, Group B) shown in the plot.
    
    Please carefully observe the time points (Time) and corresponding survival probabilities (Survival Probability) at each step-down of the curve.
    If there is a "Numbers at risk" table at the bottom, you must use it to infer the sample size at specific time points.
    
    Strictly output in JSON format, without any other explanatory text. The JSON structure should be as follows:
    {
      "Group_A": [
        {"time": 0, "survival_rate": 1.0, "at_risk": 100},
        {"time": 5, "survival_rate": 0.85, "at_risk": 85}
      ],
      "Group_B": [
        {"time": 0, "survival_rate": 1.0, "at_risk": 100},
        {"time": 4, "survival_rate": 0.60, "at_risk": 60}
      ]
    }
    """
    
try:
        # Open the uploaded image
        img = Image.open(image_file)
        
        # USE THIS UPDATED SYNTAX
        # Directly use 'gemini-1.5-flash' without any prefixes
        response = client.models.generate_content(
            model='gemini-1.5-flash-latest', 
            contents=[prompt, img]
        )

    
        # Clean the LLM output to ensure it is a valid JSON string
        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        return json.loads(result_text)
        
    except Exception as e:
        return {"error": str(e)}
