import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test

def reconstruct_patient_data(group_data):
    times = []
    events = []
    for i in range(1, len(group_data)):
        prev_node = group_data[i-1]
        curr_node = group_data[i]
        events_occurred = int((prev_node["survival_rate"] - curr_node["survival_rate"]) * prev_node["at_risk"])
        if events_occurred > 0:
            times.extend([curr_node["time"]] * events_occurred)
            events.extend([1] * events_occurred)
    return pd.DataFrame({"time": times, "event": events})

def calculate_log_rank(json_data):
    groups = list(json_data.keys())
    if len(groups) < 2:
        return "Need at least 2 groups."
    df_A = reconstruct_patient_data(json_data[groups[0]])
    df_B = reconstruct_patient_data(json_data[groups[1]])
    results = logrank_test(df_A['time'], df_B['time'], df_A['event'], df_B['event'])
    return results.p_value
