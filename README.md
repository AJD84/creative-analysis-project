# Creative Performance Analysis Platform

An AI-powered analytics tool that helps digital marketers optimize Facebook/Meta advertising campaigns by analyzing creative performance and identifying winning patterns.

## 🚀 What It Does

This platform analyzes your advertising creative data to:
- Calculate a composite Creative Score (0-100) for each ad
- Identify top and bottom performing creatives
- Generate interactive performance dashboards
- Provide AI-powered insights on what creative elements drive results
- Create actionable hypotheses for your next creative brief

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/AJD84/creative-analysis-project.git
cd creative-analysis-project
```

2. Install required dependencies:
```bash
pip install pandas numpy plotly
```

## 📖 Usage

### Step 1: Prepare Your Data

Export your campaign data from Meta Ads Manager as a CSV file and save it as `raw_creative_data.csv` in the project directory. 

**Required columns:**
- Ad name
- Ads (Ad ID)
- Creative ID
- CTR (all)
- Preview link
- Amount spent (AUD)
- Impressions
- Reach
- Frequency
- Clicks (all)
- Outbound clicks
- Purchases
- Purchase ROAS (return on ad spend)
- Video plays at 95%

### Step 2: Run the Creative Pipeline

```bash
python3 creative_pipeline.py
```

This will:
- Clean and process your raw data
- Calculate Creative Scores
- Generate `dashboard.html` with interactive visualizations
- Export `ai_correlation_data.csv` for AI analysis

### Step 3: Run AI Analysis

```bash
python3 ai_analysis.py
```

This will:
- Analyze creative patterns using **Real AI** (if OpenAI API key is set) or **Mock AI** (fallback)
- Generate actionable hypotheses based on creative elements
- Create `final_ai_creative_report.csv` with detailed tags

**For Real AI Analysis:** See [PHASE1_GUIDE.md](PHASE1_GUIDE.md) for setup instructions.

### Step 4: View Results

Open `dashboard.html` in your web browser to see:
- Top 10 and Bottom 10 performing creatives
- Interactive scatter plots (ROAS vs CPA, CTR vs CVR)
- AI-generated creative hypotheses
- Clickable links to view each creative

## 📊 Outputs

| File | Description |
|------|-------------|
| `dashboard.html` | Interactive performance dashboard |
| `ai_correlation_data.csv` | Processed data for AI analysis |
| `final_ai_creative_report.csv` | AI tags and analysis results |

## 🎯 Creative Score Calculation

The Creative Score is a weighted composite of:
- **CTR (40%)** - Click-through rate
- **CVR (30%)** - Conversion rate
- **ROAS (20%)** - Return on ad spend
- **Video Completion (10%)** - 95% video play-through rate

## ⚠️ AI Analysis Modes

The platform now supports **TWO MODES** for creative analysis:

### 🤖 Real AI Mode (Phase 1 - NEW!)
- Uses OpenAI GPT-4V for actual image/video analysis
- Requires OpenAI API key (sign up at platform.openai.com)
- Cost: ~$0.01-0.05 per creative analyzed
- Provides accurate, AI-powered creative insights

**Setup:**
```bash
pip install openai
export OPENAI_API_KEY='your-api-key-here'
python3 ai_analysis.py
```

See **[PHASE1_GUIDE.md](PHASE1_GUIDE.md)** for complete setup instructions.

### 🎭 Mock AI Mode (Fallback)
- Simulates AI analysis with randomized tags
- No API key required - works out of the box
- Free to use, instant results
- Good for testing and development

**The script automatically uses Real AI if configured, otherwise falls back to Mock AI.**

---

## 🔍 For a Complete Analysis

See **[ANALYSIS.md](ANALYSIS.md)** for:
- Detailed pros and cons for users
- Business opportunity assessment
- Improvement recommendations
- Monetization strategies
- Competitive analysis
- Go-to-market strategy
- Investment requirements

## 📈 Data Filters

The analysis applies these quality filters:
- Minimum spend: $50 AUD
- Minimum impressions: 1,000
- Must have at least 1 purchase
- Excludes DPA (Dynamic Product Ads) and dynamic sets

You can adjust these filters in the `apply_quality_filters()` function in `creative_pipeline.py`.

## 🛠️ Customization

### Adjust Score Weights

Edit the `SCORE_WEIGHTS` dictionary in `creative_pipeline.py`:

```python
SCORE_WEIGHTS = {
    'CTR_Decimal': 0.40,      # Adjust weight (must sum to 1.0)
    'CVR_Decimal': 0.30,
    'ROAS_Purchase': 0.20,
    'ThruPlay_Decimal': 0.10,
}
```

### Change Column Mapping

If your Meta export has different column names, update `COLUMN_MAPPING` in `creative_pipeline.py`.

## 🤝 Contributing

This is an open project. Feel free to:
- Report bugs via GitHub Issues
- Submit pull requests for improvements
- Share feedback and suggestions

## 📄 License

This project is available for use and modification. See repository for licensing details.

## 🆘 Support

For questions or issues:
1. Check the [ANALYSIS.md](ANALYSIS.md) document
2. Review the code comments in the Python files
3. Open an issue on GitHub

## 🔮 Future Enhancements

Planned improvements include:
- Real AI vision integration (GPT-4V/Gemini)
- Web application interface
- Multi-platform support (Google Ads, TikTok, LinkedIn)
- Historical trend analysis
- Predictive analytics
- Automated budget recommendations

---

**Made for marketers who want to understand what creative elements actually drive results.**
