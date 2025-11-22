# Phase 1 Implementation Guide

## 🚀 Phase 1: MVP Development - Real AI Vision Integration

This guide walks you through implementing **Phase 1** of the Creative Analysis Platform development roadmap.

---

## ✅ What's Been Completed

### Real AI Vision Analysis Integration
- ✅ Integrated OpenAI GPT-4V API for actual image/video analysis
- ✅ Maintained backward compatibility with mock AI mode
- ✅ Added automatic fallback if API key is not configured
- ✅ Implemented structured prompt engineering for consistent tagging
- ✅ Added error handling and rate limiting
- ✅ Cost estimation tracking (~$0.01-0.05 per creative)

---

## 🔧 Setup Instructions

### Step 1: Install Dependencies

```bash
pip install openai>=1.0.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

**Cost Information:**
- GPT-4 Vision: ~$0.01-0.05 per image analyzed
- For 35 creatives: ~$0.35-$1.75 per analysis run
- Monthly estimate (weekly runs): ~$1.40-$7.00

### Step 3: Set API Key

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

To make it permanent, add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Option B: .env File**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Then modify `ai_analysis.py` to load from .env:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 4: Run with Real AI

```bash
# Make sure API key is set
echo $OPENAI_API_KEY

# Run the pipeline first (if needed)
python3 creative_pipeline.py

# Run AI analysis with real vision AI
python3 ai_analysis.py
```

You should see:
```
✅ REAL AI MODE: Using OpenAI GPT-4V for creative analysis
   Model: gpt-4-vision-preview
   Cost estimate: ~$0.01-0.05 per creative
```

---

## 🎯 How It Works

### Automatic Mode Detection

The script automatically detects whether to use real or mock AI:

```python
# Real AI mode: If OpenAI is installed AND API key is set
if USE_REAL_AI:
    analysis_function = real_vision_ai_analysis
else:
    analysis_function = mock_vision_ai_analysis
```

### Real AI Analysis Process

1. **Sends creative link** to OpenAI GPT-4V
2. **Analyzes image/video** using vision capabilities
3. **Extracts structured tags**:
   - Format (UGC-Style Video, Studio Shoot, Static Image, etc.)
   - Setting (Indoor, Outdoor, Product Demo, etc.)
   - Dominant Colors
   - Hook (how it grabs attention)
   - Emotion (feeling it evokes)
4. **Returns JSON** with consistent tags
5. **Correlates with performance** to generate hypotheses

### Fallback Behavior

- If OpenAI library not installed → Mock mode
- If API key not set → Mock mode
- If API call fails → Falls back to mock for that creative
- If JSON parsing fails → Falls back to mock for that creative

---

## 📊 Expected Output

### Real AI Mode
```
Loaded 35 creatives for AI analysis.

✅ REAL AI MODE: Using OpenAI GPT-4V for creative analysis
   Model: gpt-4-vision-preview
   Cost estimate: ~$0.01-0.05 per creative

--- Analyzing 35 creatives ---
  Progress: 5/35 creatives analyzed
  Progress: 10/35 creatives analyzed
  ...
  Progress: 35/35 creatives analyzed

--- Generating Actionable Hypotheses from AI Tags ---

======================================================================
            🔥 FINAL ACTIONABLE CREATIVE HYPOTHESES 🔥
======================================================================
✅ **WINNING HYPOTHESIS (FORMAT):** Creatives tagged as **'UGC-Style Video'** 
achieved an average Creative Score of **78.3** (vs. 42.1), 
representing a **86% higher performance** than the average.

💡 Cost estimate for this analysis: $0.35 - $1.75
```

### Mock AI Mode
```
Loaded 35 creatives for AI analysis.

⚠️  MOCK AI MODE: Using simulated analysis (set OPENAI_API_KEY to use real AI)
   Tip: Install OpenAI library with: pip install openai

--- Analyzing 35 creatives ---
[MOCK] Analyzed Creative Name (Score: 85.2). Format: UGC-Style Video...
```

---

## 🔍 Verifying Real AI is Working

Check the output for these indicators:

1. **Startup message**: `✅ REAL AI MODE`
2. **Analysis labels**: `[REAL AI]` instead of `[MOCK]`
3. **Cost estimate**: Shows at the end
4. **Timing**: Real AI takes 15-30 seconds (mock is instant)
5. **Tags**: Should be more diverse and specific than mock

---

## 💰 Cost Management

### Monitor Costs
- Check [OpenAI Usage](https://platform.openai.com/usage) dashboard
- Set spending limits in OpenAI account settings
- Each analysis run costs $0.35-$1.75 for ~35 creatives

### Reduce Costs

1. **Use "low" detail mode** (already implemented):
   ```python
   "detail": "low"  # Cheaper, faster
   ```

2. **Analyze fewer creatives**:
   - Filter in `creative_pipeline.py` to export only top 10 + bottom 10

3. **Cache results**:
   - Save `final_ai_creative_report.csv`
   - Only re-analyze new/changed creatives

4. **Test with mock first**:
   - Develop and test with mock mode
   - Switch to real AI only for production runs

---

## 🐛 Troubleshooting

### "OpenAI not available" Error
```bash
pip install openai
```

### "OpenAI API key not found" Warning
```bash
export OPENAI_API_KEY='your-key-here'
# Or check if it's set:
echo $OPENAI_API_KEY
```

### "Invalid API Key" Error
- Check key starts with `sk-`
- Verify key is active in OpenAI dashboard
- Ensure no extra spaces/quotes in the key

### "Rate limit exceeded" Error
- Script includes 0.5s delay between calls
- If still happening, increase delay in code:
  ```python
  time.sleep(1.0)  # Increase from 0.5 to 1.0
  ```

### API Call Failures
- Check creative links are publicly accessible
- Ensure URLs point to images/videos (not login pages)
- Script automatically falls back to mock on errors

---

## 📈 Next Steps in Phase 1

Now that real AI is implemented, continue with other Phase 1 tasks:

### ✅ Completed
1. Real AI vision analysis integration

### 🚧 In Progress
2. Build basic web interface (8-12 weeks)
3. Add Meta + Google Ads support (4-6 weeks)
4. Create comprehensive documentation (partially done)
5. Beta test with 10 agencies

### 📋 To Do
- Set up web framework (Flask/FastAPI)
- Design frontend UI (React/Vue)
- Implement file upload and processing
- Add user authentication
- Deploy to cloud (AWS/Heroku)

---

## 🎓 Learning Resources

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [GPT-4V Best Practices](https://platform.openai.com/docs/guides/vision/best-practices)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Cost Optimization Tips](https://platform.openai.com/docs/guides/vision/calculating-costs)

---

## 📞 Support

- **API Issues**: [OpenAI Support](https://help.openai.com/)
- **Project Issues**: Open GitHub issue
- **Cost Questions**: Check OpenAI pricing page

---

## 🎉 Success Criteria

You've successfully completed Phase 1 AI integration when:

- ✅ Script runs in REAL AI mode (not mock)
- ✅ Creative tags are specific and accurate
- ✅ Hypotheses show meaningful patterns
- ✅ Cost per analysis is $0.35-$1.75
- ✅ Analysis completes in 15-30 seconds
- ✅ Results saved to `final_ai_creative_report.csv`

---

**Phase 1 Status**: 🟢 AI Integration Complete  
**Next Phase**: Web Interface Development  
**Timeline**: 2-3 weeks for AI → 8-12 weeks for web interface
