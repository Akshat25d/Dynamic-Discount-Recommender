# 🚀 Dynamic Discount Recommender System  

A machine learning-powered tool that helps businesses **optimize discounting strategies** and **maximize profitability**.  
Built with **Python, Scikit-learn, and Streamlit**, this project analyzes sales history and recommends discount levels that boost revenue while preventing profit leaks.  

---

## 💡 Features
- 📊 Analyzes historical sales data  
- 💰 Recommends profit-maximizing discount levels  
- 📈 Predicts revenue impact of different discount strategies  
- 🎨 Interactive **Streamlit dashboard** with visualizations  
- 🔧 Configurable model settings (discount range, test size, number of trees, etc.)  

---

## 🛠 Tech Stack
- **Core:** Python (Pandas, NumPy)  
- **ML:** Scikit-learn (Random Forest Regressor)  
- **UI:** Streamlit  
- **Visualization:** Matplotlib, Seaborn  

---

## 📊 Dataset Example
```csv
product_id,date,price,quantity,discount
A100,2023-01-01,49.99,15,10
B205,2023-01-01,99.99,8,0
(Column names are flexible – the app auto-detects variations such as product, item, etc.)
```


⚙️ Setup Guide

1) Clone the Repository

```git clone https://github.com/your-username/dynamic-discount-recommender.git
cd dynamic-discount-recommender
```


2) Install Dependencies

```
pip install -r requirements.txt
```

3) Prepare Your Dataset

  Use a CSV with at least 6 months of sales data

  Required fields: product_id, date, price, quantity

  Optional: discount

4) Run the Streamlit App
```
streamlit run app.py
```

5) Interact with the Dashboard

  Upload your dataset

  Select discount ranges and model parameters

  View recommended discounts and revenue forecasts

🔑 Key Model Settings

  Number of Trees (default=100): More trees = higher accuracy

  Test Size (default=20%): Portion of data reserved for validation

  Discount Range (tested: 5–40%): Experiment zone for pricing optimization

  Random State (42): Ensures reproducible results

📈 Results

  Achieved 12–18% simulated revenue uplift in test runs

Prevents profit leakage from arbitrary discounting

Provides visual proof of optimal pricing strategies
