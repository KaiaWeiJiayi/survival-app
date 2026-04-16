import requests
import streamlit as st

def extract_data_from_km_image(image_file):
    """
    临时自检函数：列出当前 API Key 拥有权限的所有模型
    """
    API_KEY = st.secrets["GEMINI_API_KEY"]
    
    # 尝试访问 v1beta 的模型列表接口
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    
    try:
        response = requests.get(url)
        res_json = response.json()
        
        if response.status_code == 200:
            # 如果成功，我们提取出所有的模型名字
            model_list = [m['name'] for m in res_json.get('models', [])]
            return {
                "success_message": "成功连接！你的 Key 拥有的模型列表如下：",
                "available_models": model_list,
                "full_debug_info": res_json
            }
        else:
            return {
                "error": f"连接失败 (Status {response.status_code})",
                "message": res_json.get('error', {}).get('message', '未知错误')
            }
    except Exception as e:
        return {"error": f"请求异常: {str(e)}"}
