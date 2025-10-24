import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------------------------------------------------
# 1. CONFIGURATION: COLUMN MAPPING & WEIGHTS
# ----------------------------------------------------------------------

# Map the columns from your raw Meta export to clean, simple names for Python
COLUMN_MAPPING = {
    # CRITICAL IDENTIFIERS & METRICS
    'Ad name': 'ad_name',
    'Ads': 'ad_id',
    'Creative ID': 'creative_id',
    'CTR (all)': 'CTR_Raw',
    # NEW: Creative link needed for AI analysis
    'Preview link': 'creative_link',  # <--- NEWLY ADDED COLUMN

    # Metrics from your Facebook Export
    'Amount spent (AUD)': 'Spend',
    'Impressions': 'impressions',
    'Reach': 'reach',
    'Frequency': 'frequency',
    'Clicks (all)': 'Clicks_All',
    'Outbound clicks': 'Outbound_Clicks',
    'Purchases': 'Purchases',
    'Purchase ROAS (return on ad spend)': 'ROAS_Purchase',
    'Video plays at 95%': 'Video_95_Percent',
}

# Weights for the Composite Creative Score (must total 1.0)
# This defines what 'good' creative performance means for your business.
SCORE_WEIGHTS = {
    'CTR_Decimal': 0.40,      # Ad stops the scroll and gets the click
    'CVR_Decimal': 0.30,      # Ad quality leads to a purchase
    'ROAS_Purchase': 0.20,    # Efficiency/Profitability
    'ThruPlay_Decimal': 0.10, # Video attention (if applicable)
}

# ----------------------------------------------------------------------
# 2. DATA CLEANING FUNCTIONS
# ----------------------------------------------------------------------

def load_and_clean_data(file_path, mapping):
    """Loads CSV, renames columns, converts metrics to numeric, and filters."""
    df = pd.read_csv(file_path)

    # Clean header: remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)

    # Identify metric columns to convert to numeric, handling errors
    metric_cols = [
        'Spend', 'impressions', 'Outbound_Clicks', 'Purchases', 
        'ROAS_Purchase', 'Clicks_All', 'CTR_Raw'
    ]
    for col in metric_cols:
        # Replace non-numeric with NaN, then convert to float
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where critical identifiers are missing
    df.dropna(subset=['ad_name', 'ad_id', 'Spend'], inplace=True)
    
    # Filter out Dynamic Ads (often appear as aggregated names) for core creative analysis
    df = df[~df['ad_name'].str.contains('DPA|Dynamic|Set - Sales', na=False, case=False)]
    
    return df

def calculate_derivatives(df):
    """Calculates all necessary rates (CTR, CVR, ThruPlay)."""
    
    # 1. Conversion Rate (CVR) - Purchases / Outbound Clicks
    df['CVR_Decimal'] = (df['Purchases'] / df['Outbound_Clicks']).replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Click-Through Rate (CTR) - Clean the raw percentage
    # CTR_Raw is often reported as a percentage in the raw file, so we convert it to a decimal
    df['CTR_Decimal'] = (df['CTR_Raw'] / 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Cost Per Acquisition (CPA)
    df['CPA'] = (df['Spend'] / df['Purchases']).replace([np.inf, -np.inf], 0).fillna(0)

    # 4. ThruPlay (Proxy for Video View Rate)
    # Assumes Video_95_Percent is not available in the sample, so we use a proxy 
    # based on Impressions to avoid errors if that column is entirely missing/empty
    df['ThruPlay_Decimal'] = (df['Video_95_Percent'] / df['impressions']).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def apply_quality_filters(df):
    """Applies standard performance filters to focus on relevant data."""
    
    # Filter 1: Exclude low spend (to focus on tested creatives)
    df = df[df['Spend'] >= 50]
    
    # Filter 2: Exclude low volume (to focus on statistically relevant results)
    df = df[df['impressions'] >= 1000]
    
    # Filter 3: Exclude rows with zero spend or zero purchases (cleans up ratios)
    df = df[df['Spend'] > 0]
    df = df[df['Purchases'] > 0]
    
    return df

# ----------------------------------------------------------------------
# 3. SCORING LOGIC
# ----------------------------------------------------------------------

def calculate_creative_score(df, weights):
    """Calculates a composite score based on normalized metrics."""
    
    df_score = df.copy()
    
    # Identify metrics that are 'higher is better'
    metrics = list(weights.keys())
    
    # Normalize each metric to a 0-1 scale
    for metric in metrics:
        col_name_norm = f'{metric}_norm'
        
        # Calculate normalization (Min-Max)
        max_val = df_score[metric].max()
        min_val = df_score[metric].min()
        
        if max_val == min_val:
             # Handle case where all values are the same (prevents division by zero)
            df_score[col_name_norm] = 0
        else:
            df_score[col_name_norm] = (df_score[metric] - min_val) / (max_val - min_val)

    # Apply weights to the normalized scores
    df_score['Creative_Score'] = 0
    for metric, weight in weights.items():
        col_name_norm = f'{metric}_norm'
        df_score['Creative_Score'] += df_score[col_name_norm] * weight
        
    # Scale the final score to 0-100
    max_composite = df_score['Creative_Score'].max()
    df_score['Creative_Score'] = (df_score['Creative_Score'] / max_composite) * 100
    
    return df_score.sort_values(by='Creative_Score', ascending=False)

# ----------------------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # 1. Load, Clean, and Filter Data
        df_cleaned = load_and_clean_data('raw_creative_data.csv', COLUMN_MAPPING)
        df_derived = calculate_derivatives(df_cleaned)
        df_filtered = apply_quality_filters(df_derived)

        # 2. Calculate the Final Creative Score
        df_scored = calculate_creative_score(df_filtered, SCORE_WEIGHTS)
        
        # 3. Final Preparation for Dashboard and AI
        
        # Separate Top and Bottom performers for the AI analysis (Top 20 / Bottom 20)
        composite_top = df_scored.head(20)
        composite_bottom = df_scored.tail(20)

        # Ensure we have at least 35 unique ads total (Top 20 + Bottom 15)
        # This prevents the Bottom list from overlapping too much with the Top
        if len(composite_top) + len(composite_bottom) > len(df_scored):
            # If the dataset is small, take the entire set
            ai_data_to_export = df_scored.copy()
        else:
            # Combine the two lists and drop any duplicates that might have sneaked in
            ai_data_to_export = pd.concat([composite_top, composite_bottom]).drop_duplicates(subset=['ad_name'])


        # --- 4a. EXPORT DATA FOR AI (ai_correlation_data.csv) ---
        
        # Columns needed for the next phase of AI analysis
        export_columns = [
            'ad_name', 'Creative_Score', 'Spend', 'impressions', 'frequency',
            'CTR_Decimal', 'Outbound_Clicks', 'CPA', 'ROAS_Purchase', 'CVR_Decimal', 
            'ThruPlay_Decimal', 'creative_link'  # <--- CRITICAL LINK FOR AI
        ]
        
        ai_data_to_export[export_columns].to_csv('ai_correlation_data.csv', index=False)


        # --- 4b. PREPARE DASHBOARD DATA & CHARTS ---
        
        df_dash = df_scored.copy()
        # Format Metrics for Humans (readability)
        df_dash['Creative Score (0-100)'] = df_dash['Creative_Score'].round(2)
        df_dash['CPA ($)'] = df_dash['CPA'].round(2)
        df_dash['ROAS'] = df_dash['ROAS_Purchase'].round(2)
        df_dash['CTR (%)'] = (df_dash['CTR_Decimal'] * 100).round(2)
        df_dash['CVR (%)'] = (df_dash['CVR_Decimal'] * 100).round(2)

        # 1. Efficiency Chart (CPA vs. ROAS)
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

        # 2. Acquisition Chart (CTR vs. CVR)
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

        # 3. Top 10 Table
        top_10 = df_dash.head(10)
        table_cols = ['ad_name', 'Creative Score (0-100)', 'Spend', 'ROAS', 'CPA ($)', 'CTR (%)', 'CVR (%)']
        top_10_table = top_10[table_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table', 'table-striped', 'table-hover']
        )

        # 4. Bottom 10 Table
        bottom_10 = df_dash.tail(10)
        bottom_10_table = bottom_10[table_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table', 'table-striped', 'table-hover table-danger']
        )


        # --- 4c. GENERATE SINGLE HTML DASHBOARD FILE ---
        
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
        /* FINAL FIX: Force AD NAME header to align LEFT to match text */
        table th:first-child {{ 
            text-align: left !important; 
        }}
        /* EXISTING FIX: Keep all other columns CENTERED for numerical data */
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
        print("Data saved to 'ai_correlation_data.csv' (includes creative links).")
        print("Interactive dashboard created: 'dashboard.html'.")

    except FileNotFoundError:
        print("\nERROR: raw_creative_data.csv not found.")
        print("Please ensure the file is in the same directory and named correctly.")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")