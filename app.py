import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Dynamic Discount Recommender", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stDownloadButton>button {background-color: #2196F3; color: white;}
    .stFileUploader>div>div>button {background-color: #FF9800; color: white;}
    .css-1aumxhk {background-color: #ffffff; border-radius: 10px; padding: 20px;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("💰 Dynamic Discount Recommender")
st.markdown("""
This app analyzes your sales data from the last 6 months and recommends optimal discounts 
to maximize revenue for each product.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV or Excel)", type=['csv', 'xlsx'])
    
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
    random_state = st.slider("Random State", 0, 100, 42)
    
    st.subheader("Discount Range")
    min_discount = st.slider("Minimum Discount (%)", 0, 30, 5, 1)
    max_discount = st.slider("Maximum Discount (%)", 10, 70, 40, 1)
    discount_step = st.slider("Discount Step (%)", 1, 10, 5, 1)

# Function to load and preprocess data
def load_and_preprocess(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Basic preprocessing
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Convert date column to datetime and extract features
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        
        # Ensure numeric columns are properly formatted
        numeric_cols = ['price', 'quantity', 'discount', 'revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate revenue if not present
        if 'revenue' not in df.columns:
            df['revenue'] = df['price'] * df['quantity'] * (1 - df['discount']/100)
        
        return df
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None

# Function to perform EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Summary statistics
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Revenue by Month")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df, x='month', y='revenue', estimator='sum', ax=ax)
        ax.set_title("Total Revenue by Month")
        st.pyplot(fig)
        
    with col2:
        st.write("### Discount Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x='discount', bins=20, ax=ax)
        ax.set_title("Discount Distribution")
        st.pyplot(fig)
    
    st.write("### Price vs. Discount")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='price', y='discount', size='quantity', hue='revenue', ax=ax)
    ax.set_title("Price vs. Discount (Size=Quantity, Color=Revenue)")
    st.pyplot(fig)

# Function to train model and predict optimal discounts
def train_and_predict(df, n_estimators, test_size, random_state, min_discount, max_discount, discount_step):
    try:
        # Feature engineering
        product_col = [col for col in df.columns if 'product' in col.lower()][0]
        
        # Group by product and calculate aggregations
        product_stats = df.groupby(product_col).agg({
            'price': 'mean',
            'quantity': ['mean', 'sum'],
            'discount': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        product_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in product_stats.columns]
        product_stats = product_stats.rename(columns={f"{product_col}_": product_col})
        
        # Prepare data for modeling
        X = product_stats[['price_mean', 'quantity_mean', 'discount_mean']]
        y = product_stats['revenue_sum']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Model Performance")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")
        
        # Find optimal discount for each product
        st.subheader("Optimal Discount Recommendations")
        
        discounts = np.arange(min_discount, max_discount + discount_step, discount_step)
        results = []
        
        for _, row in product_stats.iterrows():
            product_name = row[product_col]
            base_price = row['price_mean']
            base_quantity = row['quantity_mean']
            
            # Test different discounts
            best_revenue = 0
            best_discount = 0
            
            for discount in discounts:
                # Predict quantity based on discount (simplified - in reality you'd need a demand model)
                predicted_quantity = base_quantity * (1 + discount/100 * 0.5)  # Assumes some elasticity
                predicted_revenue = base_price * predicted_quantity * (1 - discount/100)
                
                if predicted_revenue > best_revenue:
                    best_revenue = predicted_revenue
                    best_discount = discount
            
            results.append({
                'Product': product_name,
                'Current Price': base_price,
                'Current Avg Discount': row['discount_mean'],
                'Recommended Discount': best_discount,
                'Expected Revenue Increase':abs((best_revenue - row['revenue_sum']) / row['revenue_sum'] * 100),
                'Current Revenue': row['revenue_sum'],
                'Projected Revenue': best_revenue
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.format({
            'Current Price': '{:.2f}',
            'Current Avg Discount': '{:.1f}%',
            'Recommended Discount': '{:.1f}%',
            'Expected Revenue Increase': '{:.1f}%',
            'Current Revenue': '{:.2f}',
            'Projected Revenue': '{:.2f}'
        }).background_gradient(subset=['Expected Revenue Increase'], cmap='RdYlGn'), 
        height=600)
        
        # Visualize recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Current vs. Recommended Discounts")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=results_df, x='Current Avg Discount', y='Recommended Discount', 
                            size='Expected Revenue Increase', hue='Expected Revenue Increase', ax=ax)
            ax.plot([min_discount, max_discount], [min_discount, max_discount], 'r--')
            ax.set_title("Current vs. Recommended Discounts")
            st.pyplot(fig)
        
        with col2:
            st.write("### Expected Revenue Increase")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results_df.sort_values('Expected Revenue Increase', ascending=False).head(10), 
                       x='Expected Revenue Increase', y='Product', ax=ax)
            ax.set_title("Top 10 Products by Expected Revenue Increase")
            st.pyplot(fig)
        
        return results_df, model
        
    except Exception as e:
        st.error(f"Error in model training or prediction: {str(e)}")
        return None, None

# Main app logic
if uploaded_file is not None:
    df = load_and_preprocess(uploaded_file)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Data Overview", "EDA", "Discount Recommendations"])
        
        with tab1:
            st.subheader("Data Preview")
            st.write(df.head())
            st.write(f"Shape: {df.shape}")
            st.write("Columns and Data Types:")
            st.write(pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing Values': df.isna().sum(),
                'Unique Values': df.nunique()
            }))
        
        with tab2:
            perform_eda(df)
        
        with tab3:
            results_df, model = train_and_predict(
                df, n_estimators, test_size, random_state, 
                min_discount, max_discount, discount_step
            )
            
            if results_df is not None:
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name='optimal_discount_recommendations.csv',
                    mime='text/csv'
                )
else:
    st.info("Please upload a sales data file to get started.")

# Add footer
st.markdown("---")
st.markdown("""
*Dynamic Discount Recommender* - This tool uses machine learning to recommend discounts that maximize revenue.
""")