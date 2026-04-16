import json
import base64
import requests
import streamlit as st

def extract_data_from_km_image(image_file):
    """
    双重保险：优先尝试 Google 官方 API，失败则自动切换至 OpenRouter。
    """
    img_bytes = image_file.getvalue()
    image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # --- 尝试路径 1: Google Gemini 官方 ---
    st.write("🔄 尝试通过 Google 官方 API 解析...")
    official_data = try_google_official(image_b64)
    
    if "error" not in official_data:
        st.success("✅ 官方 API 解析成功！")
        return official_data
    
    # 如果官方失败，输出原因并切换
    st.warning(f"⚠️ 官方 API 暂时不可用 (原因: {official_data['error']})，正在切换至 OpenRouter 备选通道...")
    
    # --- 尝试路径 2: OpenRouter 备选 ---
    openrouter_data = try_openrouter(image_b64)
    if "error" not in openrouter_data:
        st.success("✅ 备选通道解析成功！")
        return openrouter_data
    
    # 如果全失败了，返回最终错误
    return {"error": f"所有 AI 通道均不可用。OpenRouter 错误: {openrouter_data.get('error')}"}

def try_google_official(image_b64):
    """官方 API 调用逻辑"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return {"error": "缺少官方 API Key"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    prompt = "Analyze this KM curve. Extract Time, Survival Rate, At Risk. Output JSON ONLY."
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        res_json = response.json()
        if response.status_code == 200:
            content = res_json['candidates'][0]['content']['parts'][0]['text']
            return json.loads(content)
        return {"error": f"Status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def try_openrouter(image_b64):
    """OpenRouter 备选调用逻辑"""
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key: return {"error": "缺少 OpenRouter API Key"}
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "google/gemini-flash-1.5-exp:free",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Extract KM curve data to JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]}]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}
