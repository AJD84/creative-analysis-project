import pandas as pd
import random
import time
import os
import json

# Try to import OpenAI - if not available, will fall back to mock
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# Set your OpenAI API key as an environment variable:
# export OPENAI_API_KEY='your-api-key-here'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
USE_REAL_AI = OPENAI_AVAILABLE and OPENAI_API_KEY is not None

# AI Analysis prompt template
AI_ANALYSIS_PROMPT = """Analyze this advertising creative and provide structured tags.

Focus on:
1. FORMAT: What type of creative is this? (e.g., UGC-Style Video, Studio Shoot, Static Image, Carousel, Animated GFX)
2. SETTING: Where does it take place? (e.g., Indoor Fashion Shot, Outdoor Lifestyle, Product Demo, Text Overlay Only)
3. DOMINANT_COLOR: What are the primary colors? (e.g., Black/White, Vibrant Pink/Red, Muted Earth Tones, Cool Blue/Green)
4. HOOK: How does it grab attention in the first 3 seconds? (e.g., Strong Text Hook, Fast-paced editing, Direct-to-camera speaking, Product close-up)
5. EMOTION: What feeling does it evoke? (e.g., Excitement/Urgency, Calm/Luxurious, Informative, Aspirational)

Respond ONLY with valid JSON in this exact format:
{
  "format": "your answer here",
  "setting": "your answer here", 
  "dominant_color": "your answer here",
  "hook": "your answer here",
  "emotion": "your answer here"
}"""

# ----------------------------------------------------------------------
# REAL AI VISION ANALYSIS (OpenAI GPT-4V)
# ----------------------------------------------------------------------

def real_vision_ai_analysis(ad_name, creative_link, score):
    """Uses OpenAI GPT-4V to analyze a creative and extract structured tags."""
    
    if not USE_REAL_AI:
        raise RuntimeError("OpenAI not available. Set OPENAI_API_KEY environment variable.")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Call GPT-4V with vision capabilities
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": AI_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": creative_link,
                                "detail": "low"  # Use "low" for cost efficiency, "high" for better quality
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3  # Lower temperature for more consistent tagging
        )
        
        # Extract the JSON response
        content = response.choices[0].message.content
        
        # Parse JSON from the response
        # Handle cases where the AI might include markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        tags = json.loads(content)
        
        # Validate that all required keys are present
        required_keys = ['format', 'setting', 'dominant_color', 'hook', 'emotion']
        for key in required_keys:
            if key not in tags:
                tags[key] = "Unknown"
        
        analysis_text = f"[REAL AI] Analyzed {ad_name} (Score: {score:.1f}). Format: {tags['format']}. Hook: {tags['hook']}. Emotion: {tags['emotion']}."
        
        return tags, analysis_text
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse AI response for {ad_name}: {e}")
        # Fall back to mock analysis if parsing fails
        return mock_vision_ai_analysis(ad_name, creative_link, score)
    except Exception as e:
        print(f"Warning: AI analysis failed for {ad_name}: {e}")
        # Fall back to mock analysis if API call fails
        return mock_vision_ai_analysis(ad_name, creative_link, score)

# ----------------------------------------------------------------------
# MOCK VISION AI FUNCTION (Fallback)
# ----------------------------------------------------------------------

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
    analysis_text = f"[MOCK] Analyzed {ad_name} (Score: {score:.1f}). Format: {tags['format']}. Hook: {tags['hook']}. Emotion: {tags['emotion']}."
    
    return tags, analysis_text

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Load the data from the first pipeline step
        df = pd.read_csv('ai_correlation_data.csv')
        print(f"Loaded {len(df)} creatives for AI analysis.")
        
        # Check AI mode
        if USE_REAL_AI:
            print("\n✅ REAL AI MODE: Using OpenAI GPT-4V for creative analysis")
            print(f"   Model: gpt-4-vision-preview")
            print(f"   Cost estimate: ~$0.01-0.05 per creative")
            analysis_function = real_vision_ai_analysis
        else:
            print("\n⚠️  MOCK AI MODE: Using simulated analysis (set OPENAI_API_KEY to use real AI)")
            if not OPENAI_AVAILABLE:
                print("   Tip: Install OpenAI library with: pip install openai")
            analysis_function = mock_vision_ai_analysis
        
        # --- A. RUN AI ANALYSIS ON ALL CREATIVES ---
        
        results = []
        print(f"\n--- Analyzing {len(df)} creatives ---")
        
        for index, row in df.iterrows():
            # Analyze the creative using the selected AI function
            try:
                tags, analysis_text = analysis_function(
                    row['ad_name'], 
                    row['creative_link'], 
                    row['Creative_Score']
                )
                
                # Combine the AI's tags with the existing row data
                row_data = row.to_dict()
                row_data.update(tags)
                results.append(row_data)
                
                # Print progress
                if (index + 1) % 5 == 0 or index == len(df) - 1:
                    print(f"  Progress: {index + 1}/{len(df)} creatives analyzed")
                
                # Rate limiting for real AI to avoid hitting API limits
                if USE_REAL_AI:
                    time.sleep(0.5)  # Small delay between API calls
                    
            except Exception as e:
                print(f"  Error analyzing {row['ad_name']}: {e}")
                # On error, use basic fallback
                row_data = row.to_dict()
                row_data.update({
                    'format': 'Unknown',
                    'setting': 'Unknown',
                    'dominant_color': 'Unknown',
                    'hook': 'Unknown',
                    'emotion': 'Unknown'
                })
                results.append(row_data)
        
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
        
        if USE_REAL_AI:
            print("\n💡 Cost estimate for this analysis: $%.2f - $%.2f" % (len(df) * 0.01, len(df) * 0.05))
        
        # Export the final data set including the AI tags
        df_final.to_csv('final_ai_creative_report.csv', index=False)


    except FileNotFoundError:
        print("\nERROR: ai_correlation_data.csv not found.")
        print("Please ensure the creative_pipeline.py script was run successfully first.")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")