# Quick Start Guide

## For Users: Quick 5-Minute Setup

### 1️⃣ Install Dependencies (30 seconds)
```bash
pip install pandas numpy plotly
```

### 2️⃣ Add Your Data (1 minute)
- Export your Meta Ads campaign data as CSV
- Save it as `raw_creative_data.csv` in this folder

### 3️⃣ Run Analysis (30 seconds)
```bash
python3 creative_pipeline.py
python3 ai_analysis.py
```

### 4️⃣ View Results (2 minutes)
- Open `dashboard.html` in your browser
- Review top/bottom performers
- Read AI-generated hypotheses

---

## Key Questions Answered

### "What creative should I use for my next campaign?"
→ Look at the **Top 10 table** in dashboard.html

### "Which ads are wasting my budget?"
→ Look at the **Bottom 10 table** (red section)

### "What creative patterns work best?"
→ Read the **AI-Generated Creative Hypotheses** section

### "How do I improve my ROAS?"
→ Check the **ROAS vs CPA chart** - identify high ROAS, low CPA creatives

---

## Understanding Your Creative Score

**90-100**: Elite performers - Scale these immediately  
**70-89**: Strong performers - Maintain and optimize  
**50-69**: Average performers - Test variations  
**30-49**: Weak performers - Analyze and improve  
**0-29**: Poor performers - Pause or replace  

---

## Quick Wins

### ✅ Immediate Actions (Do Today)
1. **Pause bottom 10 creatives** → Save 20-30% of wasted spend
2. **Increase budget on top 3** → Typically 2-3x better ROAS
3. **Download top performers** → Share with creative team

### ✅ This Week
1. **Create 3 variations** of top creative
2. **Test hypotheses** from AI section (e.g., "UGC-style video")
3. **Set up weekly analysis** routine

### ✅ This Month
1. **Build creative library** of winners
2. **Document winning patterns** for future briefs
3. **Train team** on using the tool

---

## Common Issues

### "Script says file not found"
→ Make sure `raw_creative_data.csv` is in the same folder

### "Charts are blank"
→ Check that your data has at least 10 creatives with $50+ spend

### "AI hypotheses show warning"
→ This is normal - run `python3 ai_analysis.py` first

### "Scores seem wrong"
→ Adjust weights in `creative_pipeline.py` (line 28-33)

---

## Pro Tips

💡 **Run analysis weekly** to catch performance changes early  
💡 **Compare before/after** when testing new creative types  
💡 **Share dashboard.html** with clients (it's presentation-ready)  
💡 **Export CSV files** to Google Sheets for team collaboration  
💡 **Keep archive** of historical reports for trend analysis  

---

## Getting Help

📖 Read [ANALYSIS.md](ANALYSIS.md) for complete pros/cons/business analysis  
📖 Read [README.md](README.md) for detailed setup and usage  
🐛 Report issues on GitHub  

---

## Next Steps

After mastering the basics:
1. Integrate real AI vision API (OpenAI GPT-4V)
2. Analyze data from multiple platforms
3. Build historical performance tracking
4. Consider business opportunities in [ANALYSIS.md](ANALYSIS.md)
