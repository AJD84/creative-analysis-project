import pandas as pd
import numpy as np
import plotly.express as px
import subprocess
import os

# ----------------------------------------------------------------------
# 1. CONFIGURATION: COLUMN MAPPING & WEIGHTS
# ----------------------------------------------------------------------

COLUMN_MAPPING = {
    'Ad name': 'ad_name',
    'Ads': 'ad_id',
    'Creative ID': 'creative_id',
    'CTR (all)': 'CTR_Raw',
    'Preview link': 'creative_link',  # CRITICAL FOR CLICKABLE LINKS
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

SCORE_WEIGHTS = {
    'CTR_Decimal': 0.40,
    'CVR_Decimal': 0.30,
    'ROAS_Purchase': 0.20,
    'ThruPlay_Decimal': 0.10,
}

# ----------------------------------------------------------------------
# 2. DATA CLEANING FUNCTIONS (UNCHANGED)
# ----------------------------------------------------------------------

def load_and_clean_data(file_path, mapping):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)
    metric_cols = [
        'Spend', 'impressions', 'Outbound_Clicks', 'Purchases', 
        'ROAS_Purchase', 'Clicks_All', 'CTR_Raw'
    ]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['ad_name', 'ad_id', 'Spend'], inplace=True)
    df = df[~df['ad_name'].str.contains('DPA|Dynamic|Set - Sales', na=False, case=False)]
    return df

def calculate_derivatives(df):
    df['CVR_Decimal'] = (df['Purchases'] / df['Outbound_Clicks']).replace([np.inf, -np.inf], 0).fillna(0)
    df['CTR_Decimal'] = (df['CTR_Raw'] / 100).replace([np.inf, -np.inf], 0).fillna(0)
    df['CPA'] = (df['Spend'] / df['Purchases']).replace([np.inf, -np.inf], 0).fillna(0)
    df['ThruPlay_Decimal'] = (df['Video_95_Percent'] / df['impressions']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def apply_quality_filters(df):
    df = df[df['Spend'] >= 50]
    df = df[df['impressions'] >= 1000]
    df = df[df['Spend'] > 0]
    df = df[df['Purchases'] > 0]
    return df

def calculate_creative_score(df, weights):
    df_score = df.copy()
    metrics = list(weights.keys())
    for metric in metrics:
        col_name_norm = f'{metric}_norm'
        max_val = df_score[metric].max()
        min_val = df_score[metric].min()
        if max_val == min_val:
            df_score[col_name_norm] = 0
        else:
            df_score[col_name_norm] = (df_score[metric] - min_val) / (max_val - min_val)

    df_score['Creative_Score'] = 0
    for metric, weight in weights.items():
        col_name_norm = f'{metric}_norm'
        df_score['Creative_Score'] += df_score[col_name_norm] * weight
        
    max_composite = df_score['Creative_Score'].max()
    df_score['Creative_Score'] = (df_score['Creative_Score'] / max_composite) * 100
    return df_score.sort_values(by='Creative_Score', ascending=False)

# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR HTML
# ----------------------------------------------------------------------

def process_table_data(df):
    """Formats data and creates the clickable link column."""
    df_dash = df.copy()
    df_dash['Creative Score (0-100)'] = df_dash['Creative_Score'].round(2)
    df_dash['CPA ($)'] = df_dash['CPA'].round(2)
    df_dash['ROAS'] = df_dash['ROAS_Purchase'].round(2)
    df_dash['CTR (%)'] = (df_dash['CTR_Decimal'] * 100).round(2)
    df_dash['CVR (%)'] = (df_dash['CVR_Decimal'] * 100).round(2)

    # NEW: Create the clickable link column (HTML Link)
    def make_clickable_ad_name(row):
        # We use 'Ad Name (Click to View)' as the table header now
        return f'<a href="{row["creative_link"]}" target="_blank">{row["ad_name"]}</a>'
    
    # Apply the function to create the new column
    df_dash['Ad Name (Click to View)'] = df_dash.apply(make_clickable_ad_name, axis=1)

    return df_dash

def get_ai_hypotheses():
    """Reads the final hypotheses from the ai_analysis.py script output."""
    
    # --- IMPORTANT NOTE ON SUBPROCESS ---
    # Because we cannot reliably run a separate Python script ('ai_analysis.py') 
    # and capture its output (the hypotheses) across all user environments (like VS Code), 
    # we must rely on the user running the script first. The code below simulates 
    # capturing the output, but the user must ensure ai_analysis.py ran successfully.
    
    if os.path.exists('final_ai_creative_report.csv'):
        try:
            # Execute the script and capture its standard output (the print statements)
            # This is complex and often fails in certain environments.
            result = subprocess.run(['python3', 'ai_analysis.py'], capture_output=True, text=True, check=True)
            output = result.stdout
            
            start_tag = "FINAL ACTIONABLE CREATIVE HYPOTHESES"
            end_tag = "Analysis complete."
            
            if start_tag in output:
                start_index = output.find(start_tag)
                # Find the end of the hypothesis block
                end_index = output.find(end_tag, start_index)
                
                hypotheses_raw = output[start_index:end_index].strip()
                
                # Split and format the hypotheses for clean HTML display
                hypotheses_html = ""
                for line in hypotheses_raw.split('\n'):
                    # Skip the separator lines
                    if line.startswith('='): continue
                    
                    # Remove the emoji markers from the captured output for a clean, professional look
                    line = line.replace('✅', '').replace('❌', '').strip()
                    
                    if line:
                        # Convert **text** to <strong>text</strong> for bolding
                        line = line.replace('**', '<strong>').replace('</strong>', '</strong>')
                        hypotheses_html += f'<p class="lead" style="font-size:1.1rem;">{line}</p>'
                
                if hypotheses_html:
                    return hypotheses_html
        except Exception:
            # If the script fails to run, show a friendly warning.
            pass
            
    return (
        '<div class="alert alert-warning" role="alert">'
        '<strong>AI Hypotheses Not Available:</strong> Please ensure you have run the '
        '<strong>`python3 ai_analysis.py`</strong> script once in your terminal to generate the final strategic report.'
        '</div>'
    )


# ----------------------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # 1. Pipeline Execution
        df_cleaned = load_and_clean_data('raw_creative_data.csv', COLUMN_MAPPING)
        df_derived = calculate_derivatives(df_cleaned)
        df_filtered = apply_quality_filters(df_derived)
        df_scored = calculate_creative_score(df_filtered, SCORE_WEIGHTS)
        
        composite_top = df_scored.head(20)
        composite_bottom = df_scored.tail(20)

        # Export for AI (including the link)
        ai_data_to_export = pd.concat([composite_top, composite_bottom]).drop_duplicates(subset=['ad_name'])
        export_columns = [
            'ad_name', 'Creative_Score', 'Spend', 'impressions', 'frequency',
            'CTR_Decimal', 'Outbound_Clicks', 'CPA', 'ROAS_Purchase', 'CVR_Decimal', 
            'ThruPlay_Decimal', 'creative_link'
        ]
        ai_data_to_export[export_columns].to_csv('ai_correlation_data.csv', index=False)

        # 2. Prepare Dashboard Data & Charts
        df_dash = process_table_data(df_scored)
        hypotheses_html = get_ai_hypotheses() # Get the AI rules

        # 3. Chart Generation 
        fig_efficiency = px.scatter(df_dash, x='CPA ($)', y='ROAS', size='Spend', color='Creative Score (0-100)', hover_name='Ad Name (Click to View)', color_continuous_scale=px.colors.sequential.Inferno, title='1. Creative Efficiency: ROAS vs. CPA (Bubble size = Spend)')
        fig_efficiency.update_layout(xaxis_title="Cost Per Acquisition (AUD)", yaxis_title="Return On Ad Spend (ROAS)")
        efficiency_div = fig_efficiency.to_html(full_html=False, include_plotlyjs='cdn', div_id="efficiency_chart_div")

        fig_acquisition = px.scatter(df_dash, x='CTR (%)', y='CVR (%)', size='Spend', color='Creative Score (0-100)', hover_name='Ad Name (Click to View)', color_continuous_scale=px.colors.sequential.Viridis, title='2. Creative Acquisition: CTR vs. CVR (Bubble size = Spend)')
        fig_acquisition.update_layout(xaxis_title="Click-Through Rate (%)", yaxis_title="Conversion Rate (%)")
        acquisition_div = fig_acquisition.to_html(full_html=False, include_plotlyjs='cdn', div_id="acquisition_chart_div")

        # 4. Table Generation (Using the new clickable column)
        table_cols = ['Ad Name (Click to View)', 'Creative Score (0-100)', 'Spend', 'ROAS', 'CPA ($)', 'CTR (%)', 'CVR (%)']
        
        top_10_table = df_dash.head(10)[table_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table', 'table-striped', 'table-hover'],
            escape=False # CRITICAL: Allows the HTML link to be rendered
        )
        
        bottom_10_table = df_dash.tail(10)[table_cols].to_html(
            index=False, 
            float_format=lambda x: f'{x:.2f}',
            classes=['table', 'table-striped', 'table-hover table-danger'],
            escape=False # CRITICAL: Allows the HTML link to be rendered
        )


        # --- 5. GENERATE FINAL HTML DASHBOARD FILE ---
        
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
        /* Final Professional Table Alignment */
        table th:first-child {{ 
            text-align: left !important; 
        }}
        table td:first-child {{
            text-align: left !important;
        }}
        /* Keep all other columns CENTERED for numerical data */
        table th:nth-child(n+2), table td:nth-child(n+2) {{
            text-align: center !important; 
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-5">
        <h1 class="text-center mb-5 display-4 text-primary">Creative Performance Dashboard</h1>
        
        <div class="row mb-5">
            <div class="col-12">
                <h2>Top 10 Creatives (Ranked by Composite Score)</h2>
                <p class="text-muted">The Creative Score is a composite index (0-100) combining efficiency (ROAS) and acquisition (CTR, CVR) metrics. Click the **Ad Name** to view the creative.</p>
                {top_10_table}
            </div>
        </div>
        
        <div class="row mb-5">
            <div class="col-12">
                <h2 class="text-danger">Bottom 10 Creatives (Identify What to Pause)</h2>
                <p class="text-muted">These creatives have the lowest Composite Scores. Analyzing these against the Top 10 provides the greatest insight for the AI phase. Click the **Ad Name** to view the creative.</p>
                {bottom_10_table}
            </div>
        </div>
        
        <hr class="my-5">
        
        <div class="row mb-5">
            <div class="col-12">
                <h2 class="text-success">AI-Generated Creative Hypotheses</h2>
                <p class="text-muted">These are the strategic rules for your next creative brief, based on correlating the visual tags (Format, Hook, Emotion) against the Creative Score. The full data, including tags and links, is in <strong>final_ai_creative_report.csv</strong>.</p>
                {hypotheses_html}
            </div>
        </div>
        
        <hr class="my-5">

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
        print("1. Dashboard 'dashboard.html' updated (Professional view, Clickable Links, AI Hypotheses).")
        print("2. Data saved to 'ai_correlation_data.csv'.")
        print("3. Final AI tags and links are in 'final_ai_creative_report.csv'.")

    except FileNotFoundError:
        print("\nERROR: raw_creative_data.csv not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")