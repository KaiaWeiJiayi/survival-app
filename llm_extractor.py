import json
import base64
import requests
import streamlit as st

def extract_data_from_km_image(image_file):
    API_KEY = st.secrets["GEMINI_API_KEY"]
    
    try:
        img_bytes = image_file.getvalue()
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    
    prompt = """
    You are an expert Biostatistician digitizing a Kaplan-Meier survival curve.
    
    TASK:
    1. Identify all treatment strata/groups.
    2. EXTRACT the "Number at risk" table at the bottom. These are your exact ANCHOR points (e.g., at time 0, 20, 40, 60, 80).
    3. TRACE each curve and extract high-frequency intermediate points where the survival rate DROPS (the "steps").
    4. For intermediate curve points that do NOT have an exact "Number at risk" in the table, set "at_risk" to null.
    
    CRITICAL: You must extract at least 15-20 points per group to accurately capture the curve's shape.
    
    Output strictly in JSON format only:
    {
      "Group_A": [
        {"time": 0, "survival_rate": 1.0, "at_risk": 622}, 
        {"time": 5, "survival_rate": 0.96, "at_risk": null}, 
        {"time": 10, "survival_rate": 0.91, "at_risk": null},
        {"time": 20, "survival_rate": 0.88, "at_risk": 372}
      ]
    }
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
        }
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        res_json = response.json()

        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {res_json.get('error', {}).get('message', 'Unknown')}"}

        if 'candidates' in res_json and len(res_json['candidates']) > 0:
            content_text = res_json['candidates'][0]['content']['parts'][0]['text']
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.split("```")[1].split("```")[0].strip()
            return json.loads(content_text)
        else:
            return {"error": "AI returned empty response."}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
