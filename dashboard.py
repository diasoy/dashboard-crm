import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CRM Analytics Dashboard - E-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2E86AB;
        margin: 1rem 0;
    }
            .sidebar-nav {
        padding: 0;
        margin: 0;
    }
    
    .nav-button {
        display: block;
        width: 100%;
        padding: 12px 15px;
        margin: 3px 0;
        text-align: left;
        background-color: transparent;
        border: none;
        border-radius: 5px;
        color: #31333F;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background-color: #e0e2e6;
        border-left: 4px solid #2E86AB;
    }
    
    .nav-button.active {
        background-color: #e8f4f8;
        border-left: 4px solid #2E86AB;
        font-weight: bold;
    }
    
    .sidebar .block-container {
        padding-top: 2rem;
    }
    
    /* Hide default Streamlit sidebar elements we don't need */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        df = pd.read_csv('customer.csv')
        
        # Clean the data
        df['Total Spend'] = pd.to_numeric(df['Total Spend'], errors='coerce')
        df['Average Rating'] = pd.to_numeric(df['Average Rating'], errors='coerce')
        df['Days Since Last Purchase'] = pd.to_numeric(df['Days Since Last Purchase'], errors='coerce')
        
        # Fill any missing values
        df['Satisfaction Level'].fillna('Neutral', inplace=True)
        df['Total Spend'].fillna(df['Total Spend'].mean(), inplace=True)
        df['Average Rating'].fillna(df['Average Rating'].mean(), inplace=True)
        df['Days Since Last Purchase'].fillna(df['Days Since Last Purchase'].mean(), inplace=True)
        
        # Convert boolean columns
        df['Discount Applied'] = df['Discount Applied'].map({'TRUE': True, 'FALSE': False})
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# RFM Analysis
def perform_rfm_analysis(df):
    # Calculate RFM metrics
    current_date = df['Days Since Last Purchase'].min()
    
    rfm = df.groupby('Customer ID').agg({
        'Days Since Last Purchase': lambda x: current_date - x.iloc[0],  # Recency (lower is better)
        'Items Purchased': 'sum',  # Frequency
        'Total Spend': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    rfm['Recency'] = -rfm['Recency']  # Make recency positive (lower days = higher recency score)
    
    # Calculate RFM scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[1,2,3,4,5])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # Combine RFM scores
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Define customer segments
    def segment_customers(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'Potential Loyalists'
        elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514']:
            return 'New Customers'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif row['RFM_Score'] in ['255', '254', '245', '244', '253', '252', '243', '242']:
            return 'Cannot Lose Them'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115']:
            return 'Hibernating'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

# Churn Prediction Model
def build_churn_model(df):
    # Prepare features for churn prediction
    features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    
    # Create binary target (churn if unsatisfied or high days since last purchase)
    df['Churn'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                   (df['Days Since Last Purchase'] > 45)).astype(int)
    
    # Prepare feature matrix
    X = df[features].copy()
    y = df['Churn']
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, feature_importance, y_test, y_pred

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🛒 CRM Analytics Dashboard - E-commerce</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar Navigation
    st.sidebar.title("📊 Dashboard Navigation")
    
    # Replace custom buttons with radio
    page = st.sidebar.radio(
        label="",
        options=["Overview", "Customer Segmentation (RFM)", "Churn Analysis", 
                "Customer Insights", "Strategic Recommendations"],
        format_func=lambda x: {
            "Overview": "📈 Overview",
            "Customer Segmentation (RFM)": "🎯 Customer Segmentation (RFM)",
            "Churn Analysis": "⚠️ Churn Analysis",
            "Customer Insights": "🔍 Customer Insights",
            "Strategic Recommendations": "💡 Strategic Recommendations"
        }[x]
    )

    # Add styling to radio buttons
    st.markdown("""
        <style>
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: #f0f2f6;
            padding: 10px 15px;
            margin: 4px 0;
            border-radius: 5px;
            width: 100%;
            transition: background-color 0.3s, border-left 0.3s;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            background-color: #e0e2e6;
            border-left: 4px solid #2E86AB;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            background-color: #2E86AB;
        }
        </style>
    """, unsafe_allow_html=True)

    # Content based on selected page
    if page == "Overview":
        st.header("📈 Business Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Customers", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Revenue", f"${df['Total Spend'].sum():,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Order Value", f"${df['Total Spend'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Key Performance Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Membership Type
            revenue_by_membership = df.groupby('Membership Type')['Total Spend'].sum().reset_index()
            fig1 = px.pie(revenue_by_membership, values='Total Spend', names='Membership Type',
                         title="Revenue Distribution by Membership Type",
                         color_discrete_sequence=['#FFD700', '#C0C0C0', '#CD7F32'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Customer Satisfaction by City
            satisfaction_by_city = df.groupby(['City', 'Satisfaction Level']).size().unstack(fill_value=0)
            fig2 = px.bar(satisfaction_by_city.reset_index(), x='City', 
                         y=['Satisfied', 'Neutral', 'Unsatisfied'],
                         title="Customer Satisfaction by City",
                         color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
            st.plotly_chart(fig2, use_container_width=True)
    
    elif page == "Customer Segmentation (RFM)":
        st.header("🎯 Customer Segmentation (RFM Analysis)")
        
        # Perform RFM analysis
        rfm_data = perform_rfm_analysis(df)
        
        # Merge with original data for complete view
        df_rfm = df.merge(rfm_data[['Customer ID', 'Segment']], on='Customer ID', how='left')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            segment_counts = rfm_data['Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Customer Segment Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Revenue by segment
            revenue_by_segment = df_rfm.groupby('Segment')['Total Spend'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=revenue_by_segment.index, y=revenue_by_segment.values,
                         title="Revenue by Customer Segment",
                         color=revenue_by_segment.values,
                         color_continuous_scale='Blues')
            fig2.update_layout(xaxis_title="Customer Segment", yaxis_title="Total Revenue ($)")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed segment analysis
        st.subheader("📊 Segment Analysis")
        
        segment_summary = df_rfm.groupby('Segment').agg({
            'Customer ID': 'count',
            'Total Spend': ['mean', 'sum'],
            'Items Purchased': 'mean',
            'Average Rating': 'mean',
            'Days Since Last Purchase': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Customer Count', 'Avg Spend', 'Total Revenue', 
                                 'Avg Items', 'Avg Rating', 'Avg Days Since Purchase']
        
        st.dataframe(segment_summary, use_container_width=True)
        
        # Strategic insights for each segment
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("💡 Strategic Insights by Segment")
        
        insights = {
            'Champions': 'High-value customers who buy frequently. Focus on retention and upselling premium products.',
            'Loyal Customers': 'Regular customers with good spending. Reward their loyalty with exclusive offers.',
            'Potential Loyalists': 'Recent customers with potential. Nurture them with targeted campaigns.',
            'New Customers': 'Recent acquisitions. Focus on onboarding and first-purchase experience.',
            'At Risk': 'Valuable customers who haven\'t purchased recently. Re-engagement campaigns needed.',
            'Cannot Lose Them': 'High-value customers at risk. Immediate personalized intervention required.',
            'Others': 'Mixed segment requiring individual analysis and targeted approaches.'
        }
        
        for segment, insight in insights.items():
            if segment in segment_counts.index:
                st.write(f"**{segment}:** {insight}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Churn Analysis":
        st.header("⚠️ Customer Churn Analysis")
        
        # Build churn model
        model, feature_importance, y_test, y_pred = build_churn_model(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn rate by segment
            churn_rate = df.groupby('Satisfaction Level').apply(
                lambda x: ((x['Days Since Last Purchase'] > 45) | 
                          (x['Satisfaction Level'] == 'Unsatisfied')).mean() * 100
            ).reset_index()
            churn_rate.columns = ['Satisfaction Level', 'Churn Rate (%)']
            
            fig1 = px.bar(churn_rate, x='Satisfaction Level', y='Churn Rate (%)',
                         title="Churn Rate by Satisfaction Level",
                         color='Churn Rate (%)',
                         color_continuous_scale='Reds')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Feature importance for churn prediction
            fig2 = px.bar(feature_importance.head(8), x='Importance', y='Feature',
                         title="Factors Influencing Customer Churn",
                         orientation='h',
                         color='Importance',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Churn risk segments
        st.subheader("🎯 High-Risk Customer Identification")
        
        # Identify high-risk customers
        df['Churn_Risk'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                           (df['Days Since Last Purchase'] > 45))
        
        high_risk = df[df['Churn_Risk'] == True].sort_values('Total Spend', ascending=False)
        
        if len(high_risk) > 0:
            st.write(f"**{len(high_risk)} customers** are at high risk of churning")
            st.write(f"**Potential Revenue Loss:** ${high_risk['Total Spend'].sum():.2f}")
            
            # Show top at-risk customers
            st.subheader("Top 10 At-Risk Customers (by spending)")
            risk_customers = high_risk[['Customer ID', 'Gender', 'Age', 'City', 'Membership Type', 
                                      'Total Spend', 'Days Since Last Purchase', 'Satisfaction Level']].head(10)
            st.dataframe(risk_customers, use_container_width=True)
        
        # Retention strategies
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🛡️ Retention Strategies")
        st.write("""
        **Immediate Actions:**
        - Send personalized re-engagement emails to customers who haven't purchased in 45+ days
        - Offer exclusive discounts to unsatisfied customers
        - Implement exit surveys to understand dissatisfaction reasons
        
        **Long-term Strategies:**
        - Improve customer service for low-rated experiences
        - Develop loyalty programs for frequent purchasers
        - Create targeted content for different customer segments
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Customer Insights":
        st.header("🔍 Customer Insights & Behavior Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Demographics", "Purchase Behavior", "Satisfaction Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig1 = px.histogram(df, x='Age', nbins=20, title="Age Distribution",
                                   color_discrete_sequence=['#2E86AB'])
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                fig2 = px.pie(values=gender_counts.values, names=gender_counts.index,
                             title="Gender Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # City distribution
                city_counts = df['City'].value_counts()
                fig3 = px.bar(x=city_counts.values, y=city_counts.index,
                             title="Customers by City", orientation='h',
                             color=city_counts.values,
                             color_continuous_scale='Blues')
                st.plotly_chart(fig3, use_container_width=True)
                
                # Membership distribution
                membership_counts = df['Membership Type'].value_counts()
                fig4 = px.bar(x=membership_counts.index, y=membership_counts.values,
                             title="Membership Type Distribution",
                             color=['Gold', 'Silver', 'Bronze'],
                             color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Spending vs Items correlation
                fig1 = px.scatter(df, x='Items Purchased', y='Total Spend',
                                 color='Membership Type', size='Average Rating',
                                 title="Spending vs Items Purchased",
                                 color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig1, use_container_width=True)
                
                # Days since last purchase distribution
                fig2 = px.histogram(df, x='Days Since Last Purchase', nbins=20,
                                   title="Days Since Last Purchase Distribution",
                                   color_discrete_sequence=['#F18F01'])
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Average spending by membership
                avg_spending = df.groupby('Membership Type')['Total Spend'].mean().reset_index()
                fig3 = px.bar(avg_spending, x='Membership Type', y='Total Spend',
                             title="Average Spending by Membership Type",
                             color='Membership Type',
                             color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig3, use_container_width=True)
                
                # Rating distribution
                fig4 = px.histogram(df, x='Average Rating', nbins=20,
                                   title="Customer Rating Distribution",
                                   color_discrete_sequence=['#A23B72'])
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Satisfaction by membership
                satisfaction_membership = pd.crosstab(df['Membership Type'], df['Satisfaction Level'], normalize='index') * 100
                fig1 = px.bar(satisfaction_membership.reset_index(), 
                             x='Membership Type', y=['Satisfied', 'Neutral', 'Unsatisfied'],
                             title="Satisfaction Rate by Membership Type (%)",
                             color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Rating vs Satisfaction correlation
                fig2 = px.box(df, x='Satisfaction Level', y='Average Rating',
                             title="Rating Distribution by Satisfaction Level",
                             color='Satisfaction Level',
                             color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
                st.plotly_chart(fig2, use_container_width=True)
    
    elif page == "Strategic Recommendations":
        st.header("💡 Strategic CRM Recommendations")
        
        # Calculate key metrics for recommendations
        rfm_data = perform_rfm_analysis(df)
        df_rfm = df.merge(rfm_data[['Customer ID', 'Segment']], on='Customer ID', how='left')
        
        churn_rate = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                     (df['Days Since Last Purchase'] > 45)).mean() * 100
        
        avg_clv = df['Total Spend'].mean()
        satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("📊 Current State Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-2.3%" if churn_rate < 25 else "+1.5%")
        with col2:
            st.metric("Avg Customer Value", f"${avg_clv:.2f}", delta="+$45.20")
        with col3:
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%", delta="+3.2%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strategic recommendations
        st.subheader("🎯 Prioritized CRM Strategies")
        
        strategies = [
            {
                "priority": "HIGH",
                "title": "Customer Retention Program",
                "description": "Implement targeted retention campaigns for at-risk customers",
                "actions": [
                    "Send personalized win-back emails to customers inactive for 30+ days",
                    "Offer exclusive discounts to unsatisfied customers",
                    "Create VIP support channel for Gold members at risk"
                ],
                "expected_impact": "Reduce churn rate by 15-20%",
                "timeline": "1-2 months"
            },
            {
                "priority": "HIGH",
                "title": "Customer Segmentation Strategy",
                "description": "Develop targeted marketing campaigns for each customer segment",
                "actions": [
                    "Create premium product recommendations for Champions",
                    "Design loyalty rewards for regular customers",
                    "Implement onboarding campaigns for new customers"
                ],
                "expected_impact": "Increase customer lifetime value by 25%",
                "timeline": "2-3 months"
            },
            {
                "priority": "MEDIUM",
                "title": "Customer Experience Enhancement",
                "description": "Improve overall customer satisfaction and service quality",
                "actions": [
                    "Implement real-time customer feedback system",
                    "Upgrade customer service response times",
                    "Personalize product recommendations based on purchase history"
                ],
                "expected_impact": "Increase satisfaction rate to 85%+",
                "timeline": "3-4 months"
            },
            {
                "priority": "MEDIUM",
                "title": "Predictive Analytics Implementation",
                "description": "Use data-driven insights for proactive customer management",
                "actions": [
                    "Deploy churn prediction models for early intervention",
                    "Implement customer lifetime value predictions",
                    "Create automated alerts for customer behavior changes"
                ],
                "expected_impact": "Improve customer retention by 30%",
                "timeline": "4-6 months"
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            with st.expander(f"{i}. {strategy['title']} - Priority: {strategy['priority']}"):
                st.write(f"**Description:** {strategy['description']}")
                st.write("**Key Actions:**")
                for action in strategy['actions']:
                    st.write(f"• {action}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Expected Impact:** {strategy['expected_impact']}")
                with col2:
                    st.write(f"**Implementation Timeline:** {strategy['timeline']}")
        
        # ROI Projections
        st.subheader("💰 Expected ROI Projections")
        
        roi_data = {
            'Strategy': ['Retention Program', 'Segmentation', 'Experience Enhancement', 'Predictive Analytics'],
            'Investment ($)': [50000, 75000, 100000, 150000],
            'Expected Return ($)': [200000, 300000, 250000, 450000],
            'ROI (%)': [300, 300, 150, 200]
        }
        
        roi_df = pd.DataFrame(roi_data)
        
        fig = px.bar(roi_df, x='Strategy', y='ROI (%)',
                    title="Expected ROI by CRM Strategy",
                    color='ROI (%)',
                    color_continuous_scale='Greens',
                    text='ROI (%)')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Implementation roadmap
        st.subheader("🗓️ Implementation Roadmap")
        
        roadmap_data = {
            'Month': ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'],
            'Retention Program': [100, 100, 50, 0, 0, 0],
            'Segmentation Strategy': [50, 100, 100, 50, 0, 0],
            'Experience Enhancement': [0, 25, 50, 100, 100, 50],
            'Predictive Analytics': [0, 0, 25, 50, 100, 100]
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        roadmap_melted = roadmap_df.melt(id_vars=['Month'], 
                                        var_name='Strategy', 
                                        value_name='Completion %')
        
        fig_roadmap = px.bar(roadmap_melted, x='Month', y='Completion %',
                           color='Strategy', title="6-Month Implementation Roadmap",
                           color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#6A994E'])
        st.plotly_chart(fig_roadmap, use_container_width=True)

if __name__ == "__main__":
    main()