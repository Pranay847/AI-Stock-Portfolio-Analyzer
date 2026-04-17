# 📈 Generative AI Stock Portfolio Analyzer

The Generative AI Stock Portfolio Analyzer is an intelligent platform that analyzes real-time financial data, generates **Buy / Sell / Hold trading signals** using **Machine Learning (XGBoost)**, and explains decisions using **Large Language Models (LLMs)** through natural-language reasoning.

The app helps users understand stock movement patterns and learn **why** a model might recommend buying or selling, making it a powerful tool for both financial insights and practical ML education.

---

## ✨ Features

| Category | Description |
|----------|-------------|
| 🔄 Live Market Data | Downloads real-world stock data using Yahoo Finance API |
| 🤖 ML-Based Forecasting | Predicts future returns using XGBoost classifier |
| 📊 Feature Engineering | Moving averages, returns, volatility & price trends |
| 💹 Actionable Signals | Generates BUY / SELL / HOLD signals automatically |
| 🧠 LLM Reasoning | Converts raw numbers into human-readable explanations |
| 🖥 Interactive UI | Streamlit dashboard with charts & probability scoring |
| 📁 Multi-ticker Mode | Analyze a custom portfolio |
| 📉 Performance Metrics | Accuracy evaluation & model statistics |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Machine Learning | XGBoost, Scikit-Learn |
| Data | yfinance (Yahoo Finance) |
| Backend / Logic | Python, Pandas, NumPy |
| LLM & Reasoning | LangChain + OpenAI GPT-4 or GPT-4o-mini |
| UI | Streamlit |
| Visualization | Matplotlib / Plotly |
| Deployment (Optional) | Streamlit Cloud / Docker / HuggingFace Spaces |

---

## 🧱 System Architecture

     ┌───────────────────────────┐
               │     User enters ticker     │
               └──────────────┬─────────────┘
                              │
                              ▼
                  ┌────────────────────┐
                  │   get_data() API    │
                  │  (yfinance loader)  │
                  └────────────┬────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │  Feature Generation │
                   │ (MA10, MA50, return)│
                   └────────────┬────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  Train & Predict w/ XGBoost │
                └──────────────┬──────────────┘
                               │ prediction probs
                               ▼
                       ┌───────────────┐
                       │ Signal Engine │ BUY/SELL/HOLD
                       └───────┬───────┘
                               │
                               ▼
                   ┌─────────────────────────┐
                   │   LLM Explanation Layer  │
                   │   (LangChain + OpenAI)   │
                   └────────────┬────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │        Streamlit UI            │
                  └──────────────────────────────┘
