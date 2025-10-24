import pandas as pd
import random
import time

# --- MOCK VISION AI FUNCTION ---
# NOTE: In a real-world scenario, this function would contain the API call
# (e.g., using the 'openai' or 'google-genai' library) to GPT-4V or Gemini.
# Since we cannot make live API calls here, this function simulates the AI's output
# by randomly assigning descriptive tags based on the ad's performance score.

def mock_vision_ai_analysis(ad_name, creative_link, score):
    """Simulates a Vision AI model analyzing a creative link and providing tags."""
    
    # 1. Base Descriptive Tags (Common to all creatives)
    tags = {
        'format': random.choice(['UGC-Style Video', 'Studio Shoot', 'Static Image', 'Carousel']),
        'setting': random.choice(['Indoor Fashion Shot', 'Outdoor Lifestyle', 'Product Demo', 'Text Overlay Only']),
        'dominant_color': random.choice(['Black/White', 'Vibrant Pink/Red', 'Muted Earth Tones', 'Cool Blue/Green'])
    }
    
    # 2. Performance-Based Pattern (Simulating AI finding a winning/losing pattern)
    if score >= 80:
        # High-performing creative patterns (simulated)
        tags['hook'] = random.choice(['Strong Text Hook (5+ words)', 'Fast-paced editing', 'Direct-to-camera speaking'])
        tags['emotion'] = random.choice(['Excitement/Urgency', 'Calm/Luxurious'])
    elif score <= 30:
        # Low-performing creative patterns (simulated)
        tags['hook'] = random.choice(['Slow Intro/Weak Hook', 'Busy Background', 'No clear CTA'])
        tags['emotion'] = random.choice(['Confused/Aesthetic Only', 'Boring/Neutral'])
    else:
        # Average creative patterns
        tags['hook'] = random.choice(['Standard Product Showcase', 'Medium-paced edit'])
        tags['emotion'] = random.choice(['Informative', 'Pleasant'])

    # The AI's full analysis output
    analysis_text = f"Analyzed {ad_name} (Score: {score:.1f}). Format: {tags['format']}. Hook: {tags['hook']}. Emotion: {tags['emotion']}. Link: {creative_link[:50]}..."
    
    return tags, analysis_text

# ----------------------------------------------------------------------
# 3. MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Load the data from the first pipeline step
        df = pd.read_csv('ai_correlation_data.csv')
        print(f"Loaded {len(df)} creatives for AI analysis.")

        # --- A. RUN MOCK AI ANALYSIS ON ALL CREATIVES ---
        
        results = []
        print("\n--- Running Mock Vision AI Analysis (This would take hours with a real API) ---")
        
        for index, row in df.iterrows():
            # In a real API, the AI analyzes the URL from the 'creative_link' column
            tags, analysis_text = mock_vision_ai_analysis(
                row['ad_name'], 
                row['creative_link'], 
                row['Creative_Score']
            )
            
            # Combine the AI's tags with the existing row data
            row_data = row.to_dict()
            row_data.update(tags)
            results.append(row_data)
            
            # Print status update (Optional, shows progress)
            # print(f"  [PROCESSED] {row['ad_name']} -> Format: {tags['format']}")
        
        # Convert the results back to a DataFrame
        df_final = pd.DataFrame(results)
        
        # --- B. CORRELATION AND HYPOTHESIS GENERATION ---
        
        print("\n--- Generating Actionable Hypotheses from AI Tags ---")
        print("Comparing average Creative Score based on AI-generated tags:\n")
        
        hypotheses = []
        tag_columns = ['format', 'setting', 'dominant_color', 'hook', 'emotion']
        
        for col in tag_columns:
            # Group the data by the AI tag (e.g., 'UGC-Style Video')
            tag_summary = df_final.groupby(col)['Creative_Score'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
            
            # Filter for tags that appear at least 3 times for reliable analysis
            tag_summary = tag_summary[tag_summary['count'] >= 3] 
            
            if len(tag_summary) > 1:
                best_tag = tag_summary.iloc[0]
                worst_tag = tag_summary.iloc[-1]
                
                # Calculate the performance difference
                score_diff = (best_tag['mean'] - worst_tag['mean']) / worst_tag['mean'] * 100
                
                if score_diff > 10: # Only report significant difference (>10% better)
                    hypothesis = (
                        f"✅ **WINNING HYPOTHESIS ({col.upper()}):** Creatives tagged as **'{tag_summary.index[0]}'** "
                        f"achieved an average Creative Score of **{best_tag['mean']:.1f}** (vs. {worst_tag['mean']:.1f}), "
                        f"representing a **{score_diff:.0f}% higher performance** than the average."
                    )
                    hypotheses.append(hypothesis)
                elif score_diff < -10:
                     hypothesis = (
                        f"❌ **LOSING HYPOTHESIS ({col.upper()}):** Creatives tagged as **'{tag_summary.index[-1]}'** "
                        f"achieved an average Creative Score of **{worst_tag['mean']:.1f}**, which is "
                        f"**{abs(score_diff):.0f}% lower** than the better-performing tags."
                    )
                     hypotheses.append(hypothesis)

        # --- C. FINAL OUTPUT AND EXPORT ---
        
        print("\n" + "="*70)
        print("            🔥 FINAL ACTIONABLE CREATIVE HYPOTHESES 🔥")
        print("="*70)

        if hypotheses:
            for h in hypotheses:
                print(h)
        else:
            print("No significant performance differences (over 10%) found between AI tags.")
            
        print("\n" + "="*70)
        print("Analysis complete. Check 'final_ai_creative_report.csv' for raw data.")
        
        # Export the final data set including the AI tags
        df_final.to_csv('final_ai_creative_report.csv', index=False)


    except FileNotFoundError:
        print("\nERROR: ai_correlation_data.csv not found.")
        print("Please ensure the creative_pipeline.py script was run successfully first.")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")