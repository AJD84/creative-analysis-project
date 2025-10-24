import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURATION (FINAL, WORKING MAPPING) ---
# The left side MUST match the headers in your 'raw_creative_data.csv' exactly.
# The right side is the standardized name used in the script.
COLUMN_MAPPING = {
    # CRITICAL IDENTIFIERS & METRICS
    'Ad name': 'ad_name',           
    'Ads': 'ad_id',                 
    'Creative ID': 'creative_id',   
    'CTR (all)': 'CTR_Raw',         

    # Metrics from your Facebook Export
    'Amount spent (AUD)': 'Spend',         
    'Impressions': 'impressions',          
    'Reach': 'reach',
    'Frequency': 'frequency',
    'Clicks (all)': 'clicks',
    'Outbound clicks': 'Outbound_Clicks',  
    'Purchases': 'Conversions',             
    'Cost per result': 'CPA',              
    'Purchase ROAS (return on ad spend)': 'ROAS_Purchase',  
    'Video plays at 95%': 'ThruPlay_Raw' 
}

# --- FILTERS (EDIT THESE VALUES!) ---
MIN_SPEND = 50.0        # Exclude ads with < $50 spend
MIN_IMPRESSIONS = 1000  # Exclude ads with < 1,000 impressions
MIN_SPEND_FOR_RANKING = 50.0 
MAX_FREQUENCY = 5.0     


# --- 2. DATA LOADING & INITIAL CLEANING ---
def load_data(file_path='raw_creative_data.csv'):
    """Tries multiple encodings to reliably load CSV, renames, and cleans data."""
    
    encodings_to_try = ['utf-8', 'cp1252', 'latin-1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            print(f"Attempting to load data with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding) 
            print("Data loaded successfully!")
            break
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {file_path}. Please check the path and filename.")
            return None
        except UnicodeDecodeError:
            continue 
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}")
            return None

    if df is None:
        print("ERROR: Could not load the CSV file using any known encoding.")
        return None

    # --- Cleaning and Conversion ---
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # Check for required columns and ensure they exist (even as NaN)
    required_cols = ['ad_name', 'Spend', 'impressions', 'frequency', 'clicks', 
                     'CTR_Raw', 'Outbound_Clicks', 'Conversions', 'CPA', 
                     'ROAS_Purchase', 'ThruPlay_Raw']
                     
    for col in required_cols:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' is missing. Check your COLUMN_MAPPING.")
            # Create the column with NaN if it's missing (will be dropped later if required for ranking)
            df[col] = np.nan 

    # Clean and convert data types
    numerical_cols = ['Spend', 'impressions', 'frequency', 'clicks', 'Conversions', 'CPA', 'ROAS_Purchase', 'Outbound_Clicks']
    for col in numerical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 

    # Convert raw percentages to decimals
    if 'CTR_Raw' in df.columns:
        df['CTR_Decimal'] = pd.to_numeric(df['CTR_Raw'], errors='coerce') / 100.0
    else:
        df['CTR_Decimal'] = np.nan
    
    if 'ThruPlay_Raw' in df.columns:
        df['ThruPlay_Decimal'] = pd.to_numeric(df['ThruPlay_Raw'], errors='coerce') / 100.0
    else:
        df['ThruPlay_Decimal'] = np.nan

    # Calculate CVR 
    # Ensure Outbound_Clicks is not zero to avoid division by zero error
    df['CVR_Decimal'] = np.where(df['Outbound_Clicks'] > 0, df['Conversions'] / df['Outbound_Clicks'], 0)
    df['CVR_Decimal'].replace([np.inf, -np.inf], np.nan, inplace=True) 
    df['CVR_Decimal'].fillna(0, inplace=True) 
    
    
    # --- DIAGNOSTIC AND FILTERING ---
    initial_rows = len(df)
    
    # Apply Initial Filters (Min Spend/Impressions)
    df_clean = df[
        (df['Spend'] >= MIN_SPEND) & 
        (df['impressions'] >= MIN_IMPRESSIONS) 
    ].copy()
    
    print(f"Initial rows loaded: {initial_rows}")
    print(f"Rows dropped by minimum filters (Spend > {MIN_SPEND}, Imp > {MIN_IMPRESSIONS}): {initial_rows - len(df_clean)}")
    
    # Diagnostic check for critical missing data
    print("\n--- Mandatory Data Check ---")
    required_ranking_cols = ['Spend', 'CPA', 'ROAS_Purchase', 'CTR_Decimal', 'CVR_Decimal', 'ad_name']
    for col in required_ranking_cols:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            print(f"CRITICAL: {missing_count} rows are missing data in the '{col}' column.")
    print("---------------------------\n")

    # Final drop of rows that are missing essential data for ranking
    df_clean.dropna(subset=required_ranking_cols, inplace=True)
    
    print(f"Data successfully cleaned. {len(df_clean)} rows remaining for analysis.")
    return df_clean

# --- 3. RANKING LENS FUNCTIONS (For Terminal Output) ---

def rank_acquisition_lens(data_frame):
    # Calculates and ranks creatives based on CTR, CVR, and inverse CPA.
    df_acq = data_frame[
        (data_frame['Spend'] > MIN_SPEND_FOR_RANKING) & 
        (data_frame['frequency'] < MAX_FREQUENCY) 
    ].copy()
    
    if df_acq.empty:
        print("\n--- Running Acquisition Lens Ranking ---\nNo ads meet the Acquisition filter criteria.")
        return pd.DataFrame()

    # Normalize CPA (lower is better, so we invert the normalized score)
    max_cpa = df_acq['CPA'].max()
    df_acq['CPA_Normalized_Inverse'] = 1 - (df_acq['CPA'] / max_cpa) if max_cpa > 0 else 0

    df_acq['Acquisition_Score'] = (
        (df_acq['CTR_Decimal'] * 0.40) + 
        (df_acq['CVR_Decimal'] * 0.40) + 
        (df_acq['CPA_Normalized_Inverse'] * 0.20)
    )

    top_20 = df_acq.sort_values(by='Acquisition_Score', ascending=False).head(20)
    print("\n--- Running Acquisition Lens Ranking ---")
    print("Top 5 Acquisition Creatives:")
    print(top_20[['ad_name', 'Acquisition_Score', 'CTR_Decimal', 'CPA']].head().to_string(index=False))
    return top_20

def rank_efficiency_lens(data_frame):
    # Ranks creatives directly by ROAS after applying filters.
    df_eff = data_frame[
        (data_frame['Spend'] > MIN_SPEND_FOR_RANKING) & 
        (data_frame['frequency'] < MAX_FREQUENCY) 
    ].copy()

    if df_eff.empty:
        print("\n--- Running Efficiency Lens Ranking ---\nNo ads meet the Efficiency filter criteria.")
        return pd.DataFrame()

    top_20 = df_eff.sort_values(by='ROAS_Purchase', ascending=False).head(20)
    print("\n--- Running Efficiency Lens Ranking ---")
    print("Top 5 Efficiency Creatives (Ranked by ROAS):")
    print(top_20[['ad_name', 'ROAS_Purchase', 'Spend', 'frequency']].head().to_string(index=False))
    return top_20

def rank_composite_score(data_frame):
    # Calculates the composite creative score and finds Top/Bottom 20.
    df_comp = data_frame[
        (data_frame['Spend'] > MIN_SPEND_FOR_RANKING) & 
        (data_frame['frequency'] < MAX_FREQUENCY) 
    ].copy()

    if df_comp.empty:
        print("\n--- Running Composite Creative Score Ranking ---\nNo ads meet the Composite Score filter criteria.")
        return pd.DataFrame(), pd.DataFrame(), data_frame

    # Normalize metrics
    # Handle cases where max_roas or max_thruplay might be zero
    max_roas = df_comp['ROAS_Purchase'].max()
    max_thruplay = df_comp['ThruPlay_Decimal'].max()
    
    roas_norm = df_comp['ROAS_Purchase'] / max_roas if max_roas > 0 else 0
    thruplay_norm = df_comp['ThruPlay_Decimal'] / max_thruplay if max_thruplay > 0 else 0

    # Calculate Composite Score (Weighted Average)
    df_comp['Creative_Score'] = (
        (df_comp['CTR_Decimal'] * 0.4) + 
        (df_comp['CVR_Decimal'] * 0.3) + 
        (roas_norm * 0.2) + 
        (thruplay_norm * 0.1)
    )

    top_20 = df_comp.sort_values(by='Creative_Score', ascending=False).head(20)
    bottom_20 = df_comp.sort_values(by='Creative_Score', ascending=True).head(20)

    print("\n--- Running Composite Creative Score Ranking ---")
    print("Top 5 Composite Score Creatives:")
    print(top_20[['ad_name', 'Creative_Score', 'CTR_Decimal', 'ROAS_Purchase']].head().to_string(index=False))
    
    # Merge the new score column back into the original dataframe using ad_name (our ID)
    data_frame = data_frame.merge(df_comp[['ad_name', 'Creative_Score']], on='ad_name', how='left')
    data_frame['Creative_Score'].fillna(0, inplace=True) 
    
    return top_20, bottom_20, data_frame

# --- 4. MAIN EXECUTION BLOCK (UPDATED FOR PLOTLY EXPORT) ---
if __name__ == "__main__":
    
    df = load_data()
    
    if df is not None and not df.empty:
        # Run all three lens analyses (only for side effects like ranking/scoring)
        rank_acquisition_lens(df)
        rank_efficiency_lens(df)
        composite_top, composite_bottom, df_with_scores = rank_composite_score(df)

        # --- 4a. EXPORT DATA FOR AI (ai_correlation_data.csv) ---
        ai_data_to_export = pd.concat([composite_top, composite_bottom]).drop_duplicates(subset=['ad_name'])
        
        export_columns = [
            'ad_name', 'Creative_Score', 'Spend', 'impressions', 'frequency', 
            'CTR_Decimal', 'Outbound_Clicks', 'CPA', 'ROAS_Purchase', 'CVR_Decimal', 'ThruPlay_Decimal'
        ]
        
        final_export_cols = [col for col in export_columns if col in ai_data_to_export.columns]
        ai_data_to_export[final_export_cols].to_csv('ai_correlation_data.csv', index=False)
        
        # --- 4b. PREPARE DATA FOR PLOTLY DASHBOARD ---
        df_dash = df_with_scores.copy()
        
        # 1. Format Metrics for Humans (readability)
        max_score = df_dash['Creative_Score'].max()
        df_dash['Creative Score (0-100)'] = (df_dash['Creative_Score'] / max_score) * 100
        df_dash['CPA ($)'] = df_dash['CPA'].round(2)
        df_dash['ROAS'] = df_dash['ROAS_Purchase'].round(2)
        df_dash['CTR (%)'] = (df_dash['CTR_Decimal'] * 100).round(2)
        df_dash['CVR (%)'] = (df_dash['CVR_Decimal'] * 100).round(2)
        
        # 2. Clean Data for Charts
        df_dash.dropna(subset=['CPA ($)', 'ROAS', 'CTR (%)', 'CVR (%)'], inplace=True)
        df_dash = df_dash[(df_dash['CPA ($)'] > 0) & (df_dash['ROAS'] > 0)]
        
        # --- 4c. GENERATE PLOTLY CHARTS AND TABLES ---

        # 1. Efficiency Chart (CPA vs. ROAS) - Scatter Plot
        fig_efficiency = px.scatter(
            df_dash, 
            x='CPA ($)', 
            y='ROAS', 
            size='Spend', 
            color='Creative Score (0-100)', 
            hover_name='ad_name', 
            color_continuous_scale=px.colors.sequential.Inferno,
            title='1. Creative Efficiency: ROAS vs. CPA (Bubble size = Spend)',
        )
        fig_efficiency.update_layout(xaxis_title="Cost Per Acquisition (AUD)", yaxis_title="Return On Ad Spend (ROAS)")
        efficiency_div = fig_efficiency.to_html(full_html=False, include_plotlyjs='cdn', div_id="efficiency_chart_div")


        # 2. Acquisition Chart (CTR vs. CVR) - Scatter Plot
        fig_acquisition = px.scatter(
            df_dash, 
            x='CTR (%)', 
            y='CVR (%)', 
            size='Spend', 
            color='Creative Score (0-100)', 
            hover_name='ad_name', 
            color_continuous_scale=px.colors.sequential.Viridis,
            title='2. Creative Acquisition: CTR vs. CVR (Bubble size = Spend)',
        )
        fig_acquisition.update_layout(xaxis_title="Click-Through Rate (%)", yaxis_title="Conversion Rate (%)")
        acquisition_div = fig_acquisition.to_html(full_html=False, include_plotlyjs='cdn', div_id="acquisition_chart_div")


        # 3. Top 10 Table (Digestible Ranking) - FIX: Added 'Spend' and comma
        top_10 = df_dash.sort_values(by='Creative Score (0-100)', ascending=False).head(10)
        top_10_cols = ['ad_name', 'Creative Score (0-100)', 'Spend', 'ROAS', 'CPA ($)', 'CTR (%)', 'CVR (%)']
        top_10_table = top_10[top_10_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table table-striped table-hover']
        )
        
        # 4. Bottom 10 Table (For Diagnostic and AI Contrast) - NEW: Added Bottom 10
        bottom_10 = df_dash.sort_values(by='Creative Score (0-100)', ascending=True).head(10)
        bottom_10_cols = ['ad_name', 'Creative Score (0-100)', 'Spend', 'ROAS', 'CPA ($)', 'CTR (%)', 'CVR (%)']
        bottom_10_table = bottom_10[bottom_10_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table table-striped table-hover table-danger'] # Using table-danger for contrast
        )

        # --- 4d. GENERATE SINGLE HTML DASHBOARD FILE ---
        
        # This is the template for the website. The Python code fills in the parts
        # The entire block is now wrapped in a multi-line string (f""") to prevent the NameError.
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Creative Performance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .container-fluid {{ max-width: 1400px; }} 
        .plotly-graph-div {{ height: 500px !important; }}
        /* FINAL CSS FIX: Center all numerical columns */
        table th:nth-child(n+2), table td:nth-child(n+2) {{
            text-align: center !important; 
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-5">
        <h1 class="text-center mb-5 display-4 text-primary">Creative Performance Dashboard 🎯</h1>
        
        <div class="row mb-5">
            <div class="col-12">
                <h2>Top 10 Creatives (Ranked by Composite Score)</h2>
                <p class="text-muted">The Creative Score is a composite index (0-100) combining efficiency (ROAS) and acquisition (CTR, CVR) metrics. Use **Spend** to assess risk.</p>
                {top_10_table}
            </div>
        </div>
        
        <div class="row mb-5">
            <div class="col-12">
                <h2 class="text-danger">Bottom 10 Creatives (Identify What to Pause)</h2>
                <p class="text-muted">These creatives have the lowest Composite Scores. Analyzing these against the Top 10 provides the greatest insight for the AI phase.</p>
                {bottom_10_table}
            </div>
        </div>

        <div class="row">
            <div class="col-lg-6">
                {efficiency_div}
            </div>
            <div class="col-lg-6">
                {acquisition_div}
            </div>
        </div>

        <footer class="mt-5 text-center text-muted border-top pt-3">
            Analysis generated by creative_pipeline.py | Data is interactive: hover and zoom!
        </footer>
    </div>
</body>
</html>
"""

        with open('dashboard.html', 'w') as f:
            f.write(html_template)
        
        print("\n--- SCRIPT COMPLETE ---")
        print("Data saved to 'ai_correlation_data.csv'.")
        print("Interactive dashboard created: 'dashboard.html'.")