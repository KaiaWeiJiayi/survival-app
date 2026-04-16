import pandas as pd
import numpy as np
from lifelines.statistics import multivariate_logrank_test, logrank_test

def reconstruct_patient_data(group_data):
    """
    根据 AI 提取的 (time, survival_rate, at_risk) 序列重构个体层面数据。
    
    逻辑优化：
    1. 事件数 (Events) = 上一时刻风险人数 * (1 - 当前生存率 / 上一时刻生存率)
    2. 截尾数 (Censored) = 总减少人数 - 事件数
    """
    times = []
    events = []
    
    # 按时间升序排列，防止 AI 返回数据顺序错乱
    sorted_points = sorted(group_data, key=lambda x: x['time'])
    
    for i in range(1, len(sorted_points)):
        prev = sorted_points[i-1]
        curr = sorted_points[i]
        
        # 本时间段内的总人数变动
        total_reduction = max(0, prev["at_risk"] - curr["at_risk"])
        
        # 估算发生的死亡事件 (Events)
        if prev["at_risk"] > 0 and prev["survival_rate"] > 0:
            # 使用 Kaplan-Meier 概率公式推导死亡人数
            hazard = 1 - (curr["survival_rate"] / prev["survival_rate"])
            events_count = int(round(prev["at_risk"] * hazard))
            
            # 确保事件数不会超过总减少人数
            events_count = max(0, min(events_count, total_reduction))
        else:
            events_count = 0
            
        # 剩下的减少人数视为截尾 (Censored)
        censored_count = max(0, total_reduction - events_count)
        
        # 记录死亡样本 (Event = 1)
        times.extend([curr["time"]] * events_count)
        events.extend([1] * events_count)
        
        # 记录截尾样本 (Event = 0)
        times.extend([curr["time"]] * censored_count)
        events.extend([0] * censored_count)
            
    return pd.DataFrame({"time": times, "event": events})

def calculate_log_rank(json_data):
    """
    动态检测组别数并执行相应的 Log-rank 检验。
    支持 2 组对比或多组全局对比。
    """
    group_names = list(json_data.keys())
    all_dfs = []
    
    # 动态识别并转换所有组别
    for name in group_names:
        group_df = reconstruct_patient_data(json_data[name])
        group_df['group_name'] = name
        all_dfs.append(group_df)
    
    if not all_dfs:
        raise ValueError("未从 JSON 中解析出有效的生存数据。")
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 根据组别数量选择统计方法
    if len(group_names) == 2:
        # 两组对比 (A vs B)
        g1 = combined_df[combined_df['group_name'] == group_names[0]]
        g2 = combined_df[combined_df['group_name'] == group_names[1]]
        
        results = logrank_test(
            g1['time'], g2['time'], 
            event_observed_A=g1['event'], 
            event_observed_B=g2['event']
        )
    else:
        # 3 组及以上：执行全局 Log-rank 检验 (Global Test)
        # 对应图中显示的 p < 0.0001
        results = multivariate_logrank_test(
            combined_df['time'], 
            combined_df['group_name'], 
            event_observed=combined_df['event']
        )
    
    return results.p_value
