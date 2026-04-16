import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test

def reconstruct_patient_data(group_data):
    """
    Reconstruct individual-level survival data from LLM-extracted (time, survival_rate, at_risk).
    This is a simplified logic: if the survival rate drops and the number at risk decreases, 
    it is considered an event occurrence (Event=1).
    """
    times = []
    events = []
    
    for i in range(1, len(group_data)):
        prev_node = group_data[i-1]
        curr_node = group_data[i]
        
        # Estimate the number of events that occurred during this time interval
        events_occurred = int((prev_node["survival_rate"] - curr_node["survival_rate"]) * prev_node["at_risk"])
        
        if events_occurred > 0:
            times.extend([curr_node["time"]] * events_occurred)
            events.extend([1] * events_occurred) # 1 represents the occurrence of a terminal event
            
    return pd.DataFrame({"time": times, "event": events})

def calculate_log_rank(json_data):
    """
    Calculate the Log-rank test P-value between two groups.
    """
    groups = list(json_data.keys())
    if len(groups) < 2:
        return "Need at least 2 groups."
        
    group_A_name, group_B_name = groups[0], groups[1]
    
    # Reconstruct patient data for both groups
    df_A = reconstruct_patient_data(json_data[group_A_name])
    df_B = reconstruct_patient_data(json_data[group_B_name])
    
    # Perform the log-rank test using the correct function name
    results = logrank_test(df_A['time'], df_B['time'], event_observed_A=df_A['event'], event_observed_B=df_B['event'])
    
    return results.p_value
