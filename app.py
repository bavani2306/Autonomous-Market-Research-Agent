import streamlit as st
from tools.scraper import scrape_news
from tools.sentiment import analyze_articles
from agents.writer import generate_brief

st.set_page_config(page_title="Autonomous Market Research Studio", layout="wide")

st.title("🤖 Autonomous Multi-Agent Market Research Studio")
st.markdown("Enter any research topic, market question, or career trend below to trigger live multi-agent analysis and visual analytics.")

# Sidebar Controls
st.sidebar.header("🔬 Research Configuration")
topic = st.sidebar.text_input("Research Topic / Question", "Most preferred career in India")
num_articles = st.sidebar.slider("Articles to Analyze", 5, 25, 15)
run_button = st.sidebar.button("🚀 Run Multi-Agent Workflow")

if run_button:
    with st.spinner("Agent 1: Scraping live web signals..."):
        articles = scrape_news(topic, max_results=num_articles)
        
    if not articles:
        st.error("No articles found. Try a different query.")
    else:
        st.success(f"Successfully scraped {len(articles)} live signals!")
        
        with st.spinner("Agent 2: Running NLP Analytics & Entity Extraction..."):
            df = analyze_articles(articles,topic)
            
        # --- VISUAL ANALYTICS DASHBOARD SECTION ---
        st.markdown("---")
        st.subheader("📊 Quantitative Analytics & Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sentiment Distribution Breakdown**")
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
        with col2:
            st.markdown("**Top Domain / Category Mentions**")
            # Flatten domains for visualization
            all_doms = []
            for dstr in df['domains'].dropna():
                for d in dstr.split(", "):
                    all_doms.append(d.strip())
            import pandas as pd
            domain_series = pd.Series(all_doms).value_counts()
            st.bar_chart(domain_series)
            
        st.markdown("---")
        with st.spinner("Agent 3: Synthesizing Executive Intelligence Report..."):
            final_report = generate_brief(df, topic)
            
        st.subheader("📄 Final Generated Market Brief")
        st.markdown(final_report)
        
        with st.expander("🔍 View Raw Processed Dataframe"):
            st.dataframe(df[['title', 'domains', 'sentiment', 'score', 'link']])