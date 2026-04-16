# 5. Parse the content safely
        if 'candidates' in res_json and len(res_json['candidates']) > 0:
            content_text = res_json['candidates'][0]['content']['parts'][0]['text']
            
            # Clean possible markdown formatting
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(content_text)
        else:
            return {"error": "No data returned from AI."}
