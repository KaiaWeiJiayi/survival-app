import pandas as pd
import numpy as np
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines import CoxPHFitter
from scipy.stats import norm

def reconstruct_patient_data(group_data):
    """
    Reconstruct patient-level survival data from extracted JSON points using 
    linear interpolation for missing 'at_risk' values.
    """
    df = pd.DataFrame(group_data)
    
    # Coerce to numeric, turning 'null' or None into NaN
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['survival_rate'] = pd.to_numeric(df['survival_rate'], errors='coerce')
    df['at_risk'] = pd.to_numeric(df['at_risk'], errors='coerce')
    
    # Sort strictly by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Interpolate missing 'at risk' data linearly
    if df['at_risk'].isna().all():
        df['at_risk'] = 1000.0 # Default fallback if table is completely missing
    else:
        df['at_risk'] = df['at_risk'].interpolate(method='linear')
        df['at_risk'] = df['at_risk'].ffill().bfill()
        
    times = []
    events = []
    
    # Reconstruct individual event data step-by-step
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        n_i = prev['at_risk']
        s_prev = prev['survival_rate']
        s_curr = curr['survival_rate']
        
        # Calculate events based on survival rate drop
        if n_i > 0 and s_prev > 0 and s_prev > s_curr:
            hazard = 1 - (s_curr / s_prev)
            events_count = int(round(n_i * hazard))
        else:
            events_count = 0
            
        if events_count > 0:
            times.extend([curr['time']] * events_count)
            events.extend([1] * events_count)
        
        # Calculate censored patients
        total_drop = max(0, prev['at_risk'] - curr['at_risk'])
        censored_count = max(0, int(round(total_drop - events_count)))
        
        if censored_count > 0:
            times.extend([curr['time']] * censored_count)
            events.extend([0] * censored_count)
            
    return pd.DataFrame({"time": times, "event": events})

def calculate_log_rank(json_data):
    """
    Combine all groups and dynamically perform pairwise or multivariate Log-rank test.
    """
    all_dfs = []
    for name, points in json_data.items():
        df = reconstruct_patient_data(points)
        if not df.empty:
            df['group_name'] = name
            all_dfs.append(df)
            
    if not all_dfs:
        raise ValueError("Failed to reconstruct survival data. Please check JSON format.")
        
    combined = pd.concat(all_dfs, ignore_index=True)
    group_names = combined['group_name'].unique()
    
    if len(group_names) < 2:
        raise ValueError(f"Need at least 2 groups for comparison. Found: {len(group_names)}")
        
    if len(group_names) == 2:
        g1 = combined[combined['group_name'] == group_names[0]]
        g2 = combined[combined['group_name'] == group_names[1]]
        res = logrank_test(g1['time'], g2['time'], g1['event'], g2['event'])
    else:
        res = multivariate_logrank_test(combined['time'], combined['group_name'], combined['event'])
    
    return res.p_value, combined

# ==========================================
# EXTRA CREDIT: Bucher Method Functions
# ==========================================

def get_hr_and_se(df, target_group, ref_group):
    """
    Fit a Cox Proportional Hazards model to extract Log(HR) and Standard Error.
    """
    # Filter dataset to only include the two groups of interest
    df_subset = df[df['group_name'].isin([target_group, ref_group])].copy()
    
    if df_subset.empty or len(df_subset['group_name'].unique()) != 2:
        raise ValueError(f"Missing data for groups: {target_group} or {ref_group}")
    
    # Create a binary covariate for the Cox model (1 = Target, 0 = Reference/Control)
    df_subset['treatment_dummy'] = np.where(df_subset['group_name'] == target_group, 1, 0)
    
    # Prepare dataframe for lifelines (keep only time, event, and covariate)
    cox_data = df_subset[['time', 'event', 'treatment_dummy']]
    
    # Fit Cox PH model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='time', event_col='event')
    
    # Extract coefficients
    log_hr = cph.params_['treatment_dummy']
    se = cph.standard_errors_['treatment_dummy']
    
    return log_hr, se

def calculate_bucher_method(data1, treat_a, treat_b1, data2, treat_c, treat_b2):
    """
    Perform Indirect Treatment Comparison (A vs C) using a common comparator (B) via the Bucher method.
    Formula: ln(HR_AC) = ln(HR_AB) - ln(HR_CB)
    """
    # 1. Reconstruct DataFrames for both trials
    dfs1 = []
    for name, points in data1.items():
        df = reconstruct_patient_data(points)
        df['group_name'] = name
        dfs1.append(df)
    df_trial1 = pd.concat(dfs1, ignore_index=True)

    dfs2 = []
    for name, points in data2.items():
        df = reconstruct_patient_data(points)
        df['group_name'] = name
        dfs2.append(df)
    df_trial2 = pd.concat(dfs2, ignore_index=True)

    # 2. Fit Cox Models to get Log(HR) and SE for both trials
    # Trial 1: Treatment A vs Treatment B (Reference)
    log_hr_ab, se_ab = get_hr_and_se(df_trial1, target_group=treat_a, ref_group=treat_b1)
    
    # Trial 2: Treatment C vs Treatment B (Reference)
    log_hr_cb, se_cb = get_hr_and_se(df_trial2, target_group=treat_c, ref_group=treat_b2)

    # 3. Apply Bucher Method formulas
    # Calculate indirect Log(HR)
    log_hr_ac = log_hr_ab - log_hr_cb
    
    # Calculate combined variance and Standard Error
    var_ac = (se_ab ** 2) + (se_cb ** 2)
    se_ac = np.sqrt(var_ac)
    
    # Calculate Hazard Ratio (HR) and 95% Confidence Intervals
    hr_ac = np.exp(log_hr_ac)
    ci_lower = np.exp(log_hr_ac - 1.96 * se_ac)
    ci_upper = np.exp(log_hr_ac + 1.96 * se_ac)
    
    # Calculate P-value using Z-score
    z_score = log_hr_ac / se_ac
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    return {
            "Trial 1 (A vs B)": {"HR": np.exp(log_hr_ab), "Log_HR": log_hr_ab, "SE": se_ab},
            "Trial 2 (C vs B)": {"HR": np.exp(log_hr_cb), "Log_HR": log_hr_cb, "SE": se_cb},
            "Indirect (A vs C)": {
                "HR": hr_ac,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "P_Value": p_value,
                "Z_Score": z_score
            },
        
            "DataFrames": {
                "Trial 1": df_trial1,
                "Trial 2": df_trial2
            }
        }
