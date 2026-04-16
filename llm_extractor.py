import json
import base64
import requests
import streamlit as st
from PIL import Image

# 你的个人账号 API Key
API_KEY = st.secrets["GEMINI_API_KEY"]

def extract_data_from_km_image(image_file):
    """
    针对你的账号权限定制：使用 Gemini 3.1 Flash 进行数据提取
    """
    try:
        img_bytes = image_file.getvalue()
        image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return {"error": f"图片处理失败: {str(e)}"}

    # --- 关键改动：使用你列表中的 3.1 预览版路径 ---
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={API_KEY}"

    prompt = """
    Identify the treatment groups in this KM curve. 
    Extract Time, Survival Probability, and Numbers at Risk for each group.
    
    Output strictly in JSON format only:
    {
      "Group_A": [{"time": 0, "survival_rate": 1.0, "at_risk": 500}],
      "Group_B": [{"time": 0, "survival_rate": 1.0, "at_risk": 500}]
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

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()

        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {res_json.get('error', {}).get('message', 'Unknown')}"}

        # 根据 Gemini 3 系列的返回结构进行精准解析
        if 'candidates' in res_json and len(res_json['candidates']) > 0:
            content_text = res_json['candidates'][0]['content']['parts'][0]['text']
            # 清理可能的 markdown 标签
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.split("```")[1].split("```")[0].strip()
            return json.loads(content_text)
        else:
            return {"error": "AI response empty."}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
