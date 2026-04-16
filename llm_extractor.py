import json
import base64
import requests
import streamlit as st
from PIL import Image

# Securely fetch the API Key from Streamlit Secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

def extract_data_from_km_image(image_file):
    """
    Extract data using pure REST API (v1) to ensure maximum compatibility.
    """
    # 1. Prepare the image data
    try:
        img_bytes = image_file.getvalue()
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

    # 2. Define the REST endpoint (Using the stable v1 API)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"

    # 3. Create the prompt
    prompt = """
    You are a professional biostatistician. Please analyze this Kaplan-Meier survival curve.
    
    Tasks:
    1. Identify all treatment groups shown in the legend.
    2. For each group, extract the coordinates: Time (X-axis) and Survival Probability (Y-axis).
    3. Refer to the 'Number at risk' table to find the sample size at each time point.
    
    Output requirement:
    Strictly output in valid JSON format only, following this structure:
    {
      "Group_A": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}, {"time": 10, "survival_rate": 0.8, "at_risk": 80}],
      "Group_B": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}, {"time": 10, "survival_rate": 0.6, "at_risk": 60}]
    }
    """

    # 4. Build the payload for v1 API
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                }
            ]
        }]
    }

    # 5. Send the request
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()

        if response.status_code != 200:
            error_msg = res_json.get('error', {}).get('message', 'Unknown error')
            return {"error": f"API Error {response.status_code}: {error_msg}"}

        # 6. Parse and clean the response text
        if 'candidates' in res_json and len(res_json['candidates']) > 0:
            content_text = res_json['candidates'][0]['content']['parts'][0]['text']
            
            # Clean up potential Markdown blocks
            clean_text = content_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(clean_text)
        else:
            return {"error": "AI returned no candidates. Try a clearer image."}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
