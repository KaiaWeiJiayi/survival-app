import json
import base64
import requests
import streamlit as st
from PIL import Image

# Securely fetch the API Key
API_KEY = st.secrets["GEMINI_API_KEY"]

def extract_data_from_km_image(image_file):
    """
    Extract data using pure REST API call to bypass SDK 404 errors.
    """
    # 1. Prepare the image data
    try:
        img_bytes = image_file.getvalue()
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

    # 2. Define the REST endpoint
    # Note: Explicitly using v1 for feature support
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

    # 3. Create the prompt and payload
    prompt = """
    You are a professional biostatistician. Please analyze this Kaplan-Meier survival curve.
    Extract the treatment groups, Time points, Survival Probabilities, and Numbers at Risk.
    
    Output strictly in JSON format only:
    {
      "Group_A": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}],
      "Group_B": [{"time": 0, "survival_rate": 1.0, "at_risk": 100}]
    }
    """

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
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
        }
    }

    # 4. Send the request
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()

        if response.status_code != 200:
            error_msg = res_json.get('error', {}).get('message', 'Unknown error')
            return {"error": f"API Error {response.status_code}: {error_msg}"}

        # 5. Parse the content safely
        if 'candidates' in res_json and len(res_json['candidates']) > 0:
            content_text = res_json['candidates'][0]['content']['parts'][0]['text']
            return json.loads(content_text)
        else:
            return {"error": "No data returned from AI. Check image clarity."}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
