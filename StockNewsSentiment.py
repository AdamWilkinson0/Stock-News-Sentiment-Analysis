import streamlit as st
import pandas as pd
import datetime as dt
import io
from newsapi import NewsApiClient
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Cached resources

@st.cache_resource(show_spinner=False)
def load_finbert_pipeline(model_name: str = "yiyanghkust/finbert-tone"):
    # Caches FinBERT model - so no need to reload every search
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None)
    return nlp


@st.cache_data(ttl=600)
def fetch_company_info(ticker: str):
    # yfinance used to fetch company name using ticker
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info_raw = tk.info
        company_name = info_raw.get("longName") or info_raw.get("shortName") or ""
        info["name"] = company_name
    except Exception:
        info["name"] = ""
    return info


@st.cache_data(ttl=300)
def fetch_stock_history(ticker: str, period_days: int = 12):
    """
    Fetch daily stock history for roughly the past week.
    Returns a DataFrame indexed by date with 'Close' column.
    """
    # Fetch stock price for each day of the last two weeks, keeps fetching days until 12 trading days retrieved.
    # Used twelve to prevent data long before 14 days ago being retrieved.
    tk = yf.Ticker(ticker)
    try:
        hist = tk.history(period=f"{period_days}d", interval="1d", auto_adjust=False)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()[["Date", "Close"]]
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
        return hist
    except Exception:
        return pd.DataFrame()


def query_news_for_day(newsapi: NewsApiClient, query: str, day_date: dt.date, language: str = "en",
                       page_size: int = 100):
    # Returns list of article dictionaries
    from_date = day_date.isoformat()
    to_date = (day_date + dt.timedelta(days=1)).isoformat()
    try:
        res = newsapi.get_everything(
            q=query,
            language=language,
            from_param=from_date,
            to=to_date,
            sort_by="relevancy",
            page_size=page_size,
            page=1,
        )
        if res and res.get("status") == "ok":
            return res.get("articles", [])
        else:
            return []
    except Exception as e:
        st.warning(f"API Error {day_date}: {e}")
        return []


def article_text_for_sentiment(article: dict) -> str:
    # Forms text to be used for analysis, title & description for enough context
    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""
    combined = " ".join([title.strip(), description.strip(), content.strip()])
    return combined[:1000] # Truncate to stop the text being too long (more than 1000char)


def finbert_article_score(nlp_pipeline, texts: list):
    if not texts:
        return []
    # run in batches single requests that are too big
    # nlp works best on small batches of text, not large continuous pieces.
    all_scores = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # returns list of dicts with label & score (positive/negative/neutral, numerical score)
        res = nlp_pipeline(batch, truncation=True)
        for item in res:
            if isinstance(item, list):
                d = {entry['label'].lower(): entry['score'] for entry in item}
            elif isinstance(item, dict):
                d = {item['label'].lower(): item['score']}
            else:
                d = {}
            pos = d.get("positive", 0.0)
            neg = d.get("negative", 0.0)
            score = (pos - neg) * 5.0 # Scaled up to -5 and +5 range
            all_scores.append(float(score))
    return all_scores



# UI

st.set_page_config(page_title="Stock Sentiment & Price", layout="wide")
st.title("NYSE Stock - 14 Day News Sentiment & Price Trend")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        "Enter a NYSE stock ticker and your NewsAPI key (key . The tool will fetch news for the last 14 days and analyze sentiment.")
    ticker = st.text_input("NYSE Ticker (e.g. AAPL, JPM, MSFT)", value="AAPL", max_chars=4)
    newsapi_key = st.text_input("NewsAPI key (find at https://newsapi.org) - leave empty to skip sentiment analysis",
                                value="", type="password")
    run_button = st.button("Run analysis")

with col2:
    st.markdown("**Tools:**")
    st.markdown("Model: FinBERT (yiyanghkust/finbert-tone)")
    st.markdown("Graphs: Plotly")
    st.markdown("News & Stock Data: NewsAPI, yFinance")
    st.markdown("Data Organisation & Calculations: Pandas, NumPy")

if run_button:
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("Please enter a ticker.")
    else:
        with st.spinner("Loading FinBERT model...(first load will take longer)"):
            try:
                nlp = load_finbert_pipeline()
            except Exception as e:
                st.error(f"Error loading FinBERT model: {e}")
                st.stop()

        # Company Info
        with st.spinner("Fetching company news & stock price..."):
            info = fetch_company_info(ticker)
            company_name = info.get("name") or ""
            if company_name:
                st.write(f"Detected company - valid ticker `{ticker}`: **{company_name}**")
            else:
                st.write(f"No company name found for `{ticker}` - check this is the correct ticker")

            hist = fetch_stock_history(ticker, period_days=12)
            if hist.empty:
                st.warning("Error finding stock data for ticker")

        # Date range
        today = dt.date.today()
        dates = [today - dt.timedelta(days=i) for i in range(13, -1, -1)]
        sentiment_rows = []
        all_headlines = []
        total_articles = 0

        if not newsapi_key:
            st.info("No API key provided. Provide key for sentiment analysis.")
            for d in dates:
                sentiment_rows.append({"date": d, "sentiment": np.nan, "n_articles": 0})
        else:
            newsapi = NewsApiClient(api_key=newsapi_key)
            progress_bar = st.progress(0)
            for idx, single_day in enumerate(dates):
                progress_bar.progress((idx + 1) / len(dates))
                query = f"\"{company_name}\" OR {ticker}" if company_name else ticker
                articles = query_news_for_day(newsapi, query=query, day_date=single_day, language="en", page_size=100)
                texts = [txt for art in articles if (txt := article_text_for_sentiment(art).strip())]
                n_articles_day = len(texts)
                total_articles += n_articles_day

                if n_articles_day == 0:
                    sentiment_rows.append({"date": single_day, "sentiment": np.nan, "n_articles": 0})
                else:
                    try:
                        scores = finbert_article_score(nlp, texts)
                        daily_avg = float(np.mean(scores)) if scores else float("nan")
                        sentiment_rows.append(
                            {"date": single_day, "sentiment": daily_avg, "n_articles": n_articles_day})

                        for art, sc in zip(articles, scores):
                            all_headlines.append({
                                "date": single_day,
                                "headline": art.get("title", ""),
                                "description": art.get("description", ""),
                                "source": (art.get("source") or {}).get("name", ""),
                                "url": art.get("url", ""),
                                "sentiment": sc
                            })
                    except Exception as e:
                        sentiment_rows.append({"date": single_day, "sentiment": float("nan"), "n_articles": 0})

            progress_bar.empty()
            st.write(f"Analysed news for {len(dates)} days - articles processed: {total_articles}")

        # Streamlit dataFrame for sentiment
        s_df = pd.DataFrame(sentiment_rows)
        s_df["date"] = pd.to_datetime(s_df["date"])
        s_df = s_df.sort_values("date")
        s_df.set_index("date", inplace=True)

        avg_sent = s_df["sentiment"].mean(skipna=True)
        avg_display = avg_sent if not np.isnan(avg_sent) else None

        if not hist.empty:
            hist_df = hist.copy()
            hist_df["Date"] = pd.to_datetime(hist_df["Date"])
            hist_df = hist_df.set_index("Date")
            price_series = hist_df["Close"].sort_index()
        else:
            price_series = pd.Series(dtype=float)

        # Pearson Correlation Calculation
        correlation = np.nan
        if not price_series.empty and not s_df['sentiment'].dropna().empty:
            combined_df = pd.concat([s_df['sentiment'], price_series], axis=1)
            combined_df.columns = ['sentiment', 'price']
            cleaned_df = combined_df.dropna()
            if len(cleaned_df) >= 2:
                correlation = cleaned_df['sentiment'].corr(cleaned_df['price'])

        # Averages
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            if avg_display is None:
                st.metric(label="14-day average sentiment", value="N/A")
            else:
                st.metric(label="14-day average sentiment", value=f"{avg_display:.3f}")
                if avg_display > 1.5:
                    st.success("Overall positive sentiment")
                elif avg_display < -1.5:
                    st.error("Overall negative sentiment")
                else:
                    st.info("Overall neutral / mixed sentiment")

        with met_col2:
            if np.isnan(correlation):
                st.metric(label="Sentiment/Price Correlation", value="N/A")
                st.caption("Not enough overlapping data to calculate.")
            else:
                st.metric(label="Sentiment/Price Correlation", value=f"{correlation:.3f}")
                if correlation > 0.7:
                    st.success("Strong positive correlation")
                elif correlation > 0.3:
                    st.info("Moderate positive correlation")
                elif correlation < -0.7:
                    st.error("Strong negative correlation")
                elif correlation < -0.3:
                    st.warning("Moderate negative correlation")
                else:
                    st.info("Weak / No correlation")

        # Plotly chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=s_df.index,
                y=s_df["sentiment"],
                mode="lines+markers",
                name="Sentiment",
                hovertemplate="%{x|%Y-%m-%d}<br>Sentiment: %{y:.3f}<br>Articles: %{customdata}",
                customdata=np.stack([s_df["n_articles"].fillna(0).astype(int)], axis=-1),
            ),
            secondary_y=False,
        )
        if not price_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=price_series.index,
                    y=price_series.values,
                    mode="lines+markers",
                    name=f"{ticker} Close",
                    hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}",
                ),
                secondary_y=True,
            )
        fig.update_layout(
            title_text=f"{ticker} — Sentiment (last 14 days) vs Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=80, b=40),
            template="plotly_white",
            height=520,
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Sentiment (-5 to +5)", range=[-5.5, 5.5], secondary_y=False)
        if not price_series.empty:
            fig.update_yaxes(title_text="Close Price (USD)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.markdown("### Daily sentiment & article counts")
        display_df = s_df.copy()
        display_df.index = display_df.index.date
        display_df["sentiment"] = display_df["sentiment"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        st.dataframe(
            display_df.reset_index().rename(columns={"index": "date", "n_articles": "articles"})[
                ["date", "sentiment", "articles"]]
        )

        st.markdown("### Export Data")
        if len(all_headlines) > 0:
            headlines_df = pd.DataFrame(all_headlines)
            headlines_df = headlines_df.sort_values("date")
            csv_buffer = io.StringIO()
            headlines_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📄 Download retrieved data as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{ticker}-headlines-14days.csv",
                mime="text/csv"
            )
            st.caption("CSV includes date, headline, description, source, URL, and sentiment score")
        else:
            st.info("No headlines found to export.")

        st.markdown("### Pearson Correlation Coefficient")
        st.caption("Used to help mathematically understand how consistently the price & sentiment move together by calculating the strength and direction of their linear relationship.")
        st.markdown(
            """
            **Considerations** *- A low coefficient does not necessarily mean a weak link between the two variables.*
            - **Potential for Positive Bias:** Must consider that some articles report on recent stock price movements, which means sentiment can be a *reaction* to price changes & not just a predictor of them.
            - **Only Measures Trend Similarity:** The coefficient only takes into account how similar the graph trends are. Therefore, a consistently high but flat sentiment paired with a consistently growing stock price will have a low correlation, despite having a clear relationship.
            """
        )



        st.markdown(
            """
            ### Other Notes & Considerations
            - Sentiment reported as N/A if no articles found.
            - NewsAPI free tier has rate limits & limited coverage - so limited use and limited articles to analyse - resulting in sentiment being inaccurate on smaller companies due to limited news coverage.
            - The FinBERT model is trained for financial data, however can make mistakes as only analysing article title and description which it can misinterpret.
            - Would be more accurate to aggregate data from various sources e.g. Social Media, News, Forums.
            """
        )