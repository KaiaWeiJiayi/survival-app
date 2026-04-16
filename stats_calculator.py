import pandas as pd
import numpy as np
from lifelines.statistics import multivariate_logrank_test, logrank_test

def reconstruct_patient_data(group_data):
    df = pd.DataFrame(group_data)
    
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['survival_rate'] = pd.to_numeric(df['survival_rate'], errors='coerce')
    df['at_risk'] = pd.to_numeric(df['at_risk'], errors='coerce')
    
    df = df.sort_values('time').reset_index(drop=True)
    
    if df['at_risk'].isna().all():
        df['at_risk'] = 1000.0
    else:
        df['at_risk'] = df['at_risk'].interpolate(method='linear')
        df['at_risk'] = df['at_risk'].ffill().bfill()
        
    times = []
    events = []
    
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        n_i = prev['at_risk']
        s_prev = prev['survival_rate']
        s_curr = curr['survival_rate']
        
        if n_i > 0 and s_prev > 0 and s_prev > s_curr:
            hazard = 1 - (s_curr / s_prev)
            events_count = int(round(n_i * hazard))
        else:
            events_count = 0
            
        if events_count > 0:
            times.extend([curr['time']] * events_count)
            events.extend([1] * events_count)
        
        total_drop = max(0, prev['at_risk'] - curr['at_risk'])
        censored_count = max(0, int(round(total_drop - events_count)))
        
        if censored_count > 0:
            times.extend([curr['time']] * censored_count)
            events.extend([0] * censored_count)
            
    return pd.DataFrame({"time": times, "event": events})

def calculate_log_rank(json_data):
    all_dfs = []
    for name, points in json_data.items():
        df = reconstruct_patient_data(points)
        if not df.empty:
            df['group_name'] = name
            all_dfs.append(df)
            
    if not all_dfs:
        raise ValueError("无法重构生存数据，请检查提取的 JSON 格式。")
        
    combined = pd.concat(all_dfs, ignore_index=True)
    group_names = combined['group_name'].unique()
    
    if len(group_names) < 2:
        raise ValueError(f"需要至少 2 个治疗组进行对比，当前仅解析出 {len(group_names)} 组。")
        
    if len(group_names) == 2:
        g1 = combined[combined['group_name'] == group_names[0]]
        g2 = combined[combined['group_name'] == group_names[1]]
        res = logrank_test(g1['time'], g2['time'], g1['event'], g2['event'])
    else:
        res = multivariate_logrank_test(combined['time'], combined['group_name'], combined['event'])
    
    return res.p_value
