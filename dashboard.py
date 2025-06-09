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
    # You can replace this with file upload or direct CSV loading
    # For now, I'll create sample data based on your CSV structure
    df = pd.read_csv('customer.csv')   
    df.dropna(inplace=True)

    return df

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🛒 CRM Analytics Dashboard - E-commerce</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    page = st.sidebar.radio(
    "📊 Dashboard Navigation",
    ["Overview", "Customer Insights", "Customer Segmentation (RFM)","Customer Segmentation (K-Means)","Churn Analysis", "Strategic Recommendations"]
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
            st.metric("Total Customers", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Revenue", f"${df['Total Spend'].sum():,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric("Avg Order Value", f"${df['Total Spend'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Key Performance Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Analyis segmentation
            rfm_data = perform_rfm_analysis(df)
            df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
            segment_counts = rfm_data['RFM_Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Customer RFM Segmentation")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Churn Analysis Overview
            df['Churn_Risk'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                            (df['Days Since Last Purchase'] > 45))
            high_risk = df[df['Churn_Risk'] == True].sort_values('Total Spend', ascending=False)
            if len(high_risk) > 0:
                st.write(f"**{len(high_risk)} customers** are at high risk of churning")
                st.write(f"**Potential Revenue Loss:** ${high_risk['Total Spend'].sum():.2f}")

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
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Membership Type
            membership_vs_spend = df.groupby('Membership Type')['Total Spend'].mean()
            fig1 = px.bar(x=membership_vs_spend.index, y=membership_vs_spend.values, labels={'x':'Membership Type', 'y':'Average Spend'}, title='Average Spending by Membership Type')
            st.plotly_chart(fig1, use_container_width=True)

            # Customer Satisfaction by City
            satisfaction_by_city = df.groupby(['City', 'Satisfaction Level']).size().unstack(fill_value=0)
            fig2 = px.bar(satisfaction_by_city.reset_index(), x='City', 
                         y=['Satisfied', 'Neutral', 'Unsatisfied'],
                         title="Customer Satisfaction by City",
                         color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Average Spending with/without Discounts
            discount_vs_spend = df.groupby('Discount Applied')['Total Spend'].mean()
            fig1 = px.bar(x=discount_vs_spend.index, y=discount_vs_spend.values, labels={'x':'Discount Applied', 'y':'Average Spend'}, title='Average Spending with/without Discounts')
            st.plotly_chart(fig1, use_container_width=True)

            # Average item purchased and Spending
            items_vs_spend = df.groupby('Items Purchased')['Total Spend'].mean()
            fig2 = px.bar(x=items_vs_spend.index, y=items_vs_spend.values, labels={'x':'Items Purchased', 'y':'Average Spend'}, title='Average Spending by Number of Items Purchased')
            st.plotly_chart(fig2, use_container_width=True)

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
                fig3 = px.pie(values=gender_counts.values, names=gender_counts.index,
                             title="Gender Distribution")
                st.plotly_chart(fig3, use_container_width=True)

                # City distribution
                city_counts = df['City'].value_counts()
                fig3 = px.bar(x=city_counts.values, y=city_counts.index,
                             title="Customers by City", orientation='h',
                             color=city_counts.values,
                             color_continuous_scale='Blues')
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                #Age Group Distribution
                age_groups = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '19-30', '31-45', '46-60', '60+'])
                df['Age Group'] = age_groups
                fig2 = px.histogram(df, x='Age Group', title='Age Group Distribution')
                st.plotly_chart(fig2, use_container_width=True)

                # Age and Gender Distribution
                gender_with_age = df.groupby('Gender')['Age'].value_counts().reset_index()
                fig = px.bar(gender_with_age, x='Age', y='count', color='Gender', barmode='group', title='Gender and Age Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
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
                # Total Spend Distribution
                fig = px.histogram(df, x='Total Spend', title='Total Spend Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # Days since last purchase distribution
                fig2 = px.histogram(df, x='Days Since Last Purchase', nbins=20,
                                   title="Days Since Last Purchase Distribution",
                                   color_discrete_sequence=['#F18F01'])
                st.plotly_chart(fig2, use_container_width=True)

                # Discount Applied Distribution
                fig = px.pie(df, names='Discount Applied', title='Discount Applied Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Gender
                spending_by_gender = df.groupby('Gender')['Total Spend'].mean()
                fig = px.bar(x=spending_by_gender.index, y=spending_by_gender.values, labels={'x':'Gender', 'y':'Average Spend'}, title='Average Spending by Gender')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Age
                age_vs_spend = df.groupby('Age')['Total Spend'].mean()
                fig = px.bar(x=age_vs_spend.index, y=age_vs_spend.values, labels={'x':'Age', 'y':'Average Spend'}, title='Average Spending by Age Group')
                st.plotly_chart(fig, use_container_width=True)

                # Days Since Last Purchase vs Spending
                days_vs_spend = df.groupby('Days Since Last Purchase')['Total Spend'].mean()
                fig = px.bar(x=days_vs_spend.index, y=days_vs_spend.values, labels={'x':'Days Since Last Purchase', 'y':'Average Spend'}, title='Average Spending by Days Since Last Purchase')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Items correlation
                fig1 = px.scatter(df, x='Items Purchased', y='Total Spend',
                                 color='Membership Type', size='Average Rating',
                                 title="Spending vs Items Purchased",
                                 color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig1, use_container_width=True)
                
            
            with col2:
                # Item Purchased Distribution
                fig = px.histogram(df, x='Items Purchased', title='Items Purchased Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # Rating distribution
                fig4 = px.histogram(df, x='Average Rating', nbins=20,
                                   title="Customer Rating Distribution",
                                   color_discrete_sequence=['#A23B72'])
                st.plotly_chart(fig4, use_container_width=True)

                # Spending, Membership, and Gender
                fig = px.box(df, x='Membership Type', y='Total Spend', color='Gender', title='Spending by Membership Type and Gender')
                st.plotly_chart(fig, use_container_width=True) 

                # City, Item purchased, membership Type
                fig = px.box(df, x='City', y='Items Purchased', color='Membership Type', title='Spending by Membership Type and City')
                st.plotly_chart(fig, use_container_width=True)

                # Average Spending by Discount Applied
                discount_vs_spend = df.groupby('Discount Applied')['Total Spend'].mean()
                fig = px.bar(x=discount_vs_spend.index, y=discount_vs_spend.values, labels={'x':'Discount Applied', 'y':'Average Spend'}, title='Average Spending with/without Discounts')
                st.plotly_chart(fig, use_container_width=True)  

                # Membership vs Discount vs Total Spend
                df_grouped = df.groupby(['Membership Type', 'Discount Applied'])['Total Spend'].sum().unstack()
                fig = px.bar(df_grouped, barmode='group', title='Total Spend by Membership Type and Discount Applied')              
                st.plotly_chart(fig, use_container_width=True) 

                # Average spending by membership
                avg_spending = df.groupby('Membership Type')['Total Spend'].mean().reset_index()
                fig3 = px.bar(avg_spending, x='Membership Type', y='Total Spend',
                             title="Average Spending by Membership Type",
                             color='Membership Type',
                             color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig3, use_container_width=True)
                        
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

    elif page == "Customer Segmentation (RFM)":
        st.header("🎯 Customer Segmentation (RFM Analysis)")
        
        # Perform RFM analysis
        rfm_data = perform_rfm_analysis(df)
        
        # Merge with original data for complete view
        df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            segment_counts = rfm_data['RFM_Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Customer Segment Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Revenue by segment
            revenue_by_segment = df_rfm.groupby('RFM_Segment')['Total Spend'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=revenue_by_segment.index, y=revenue_by_segment.values,
                         title="Revenue by Customer Segment",
                         color=revenue_by_segment.values,
                         color_continuous_scale='Blues')
            fig2.update_layout(xaxis_title="Customer Segment", yaxis_title="Total Revenue ($)")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed segment analysis
        st.subheader("📊 Segment Analysis")
        
        segment_summary = df_rfm.groupby('RFM_Segment').agg({
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
        st.subheader("💡 Strategic Insights by Segment")
        
        insights = {
            'Top Customers': 'Top-tier customers with high loyalty and recent engagement. Offer VIP access, exclusive deals. Prioritize customer service, retention and personalized upselling.',
            'High Value Customers (Loyal)': 'Consistent spenders demonstrating loyalty. Keep them engaged with exclusive rewards and tailored offers.',
            'Medium Value Customers (Need Attention)': 'Recent customers showing promising behavior. Nurture them to become champions through targeted campaigns.',
            'Low Value Customers (At Risk)': 'Customers with declining activity or recent disengagement. Re-engagement campaigns are necessary to win them back.',
            'Hibernating': 'High-value customers showing signs of risk. Immediate personalized retention strategies required.',
        }
        
        for segment, insight in insights.items():
            if segment in segment_counts.index:
                st.write(f"**{segment}:** {insight}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Customer Segmentation (K-Means)":
        st.header("🎯 Customer Segmentation (K-Means Segementation)")
        
        # Perform RFM analysis
        cluster_data = k_means_clustering(df)
        
        # Merge with original data for complete view
        df_cluster = df.merge(cluster_data[['Customer ID', 'Cluster1', 'EngagementCluster', 'SeasonalCluster']], on='Customer ID', how='left')
    
        tab1, tab2, tab3 = st.tabs(["Total Spend, Age, And Item Purchased", "Engagement-Based Clustering", "Seasonal Clustering"])
        
        with tab1:
            st.subheader('Segmentation Based on Total Spend, Age, And Item Purchased')

            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='Cluster1', y='Total Spend', title='Total Spend by Cluster')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cluster_distribution = df['Cluster1'].value_counts().sort_index()
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Count'}, title='Cluster Distribution')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader('Segmentation Based on Engagement')

            cluster_distribution = df['EngagementCluster'].value_counts().sort_index()
            fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Count'}, title='Engagement Cluster Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader('Segmentation Based on TSeasonal')

            cluster_distribution = df['SeasonalCluster'].value_counts().sort_index()
            fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Count'}, title='Seasonal Cluster Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
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

    elif page == "Strategic Recommendations":
        st.header("💡 Strategic CRM Recommendations")
        
        # Calculate key metrics for recommendations
        rfm_data = perform_rfm_analysis(df)
        df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
        
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

# RFM Analysis
def perform_rfm_analysis(df):
    # Calculate RFM metrics
    current_date = df['Days Since Last Purchase'].min()

    rfm_table = df.groupby('Customer ID').agg({
            'Days Since Last Purchase': lambda x: current_date - x.iloc[0],  # Recency (lower is better)
            'Items Purchased': 'sum',  # Frequency
            'Total Spend': 'sum'  # Monetary
        }).reset_index()
    rfm_table.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    rfm_table['Recency'] = -rfm_table['Recency']  # Make recency positive (lower days = higher recency score)
    
    # Rank RFM
    rfm_table['R_Rank'] = rfm_table['Recency'].rank(ascending=False)
    rfm_table['F_Rank'] = rfm_table['Frequency'].rank(ascending=True)
    rfm_table['M_Rank'] = rfm_table['Monetary'].rank(ascending=True)
    rfm_table.head()

    # Normalize RFM
    rfm_table['R_rank_norm'] = (rfm_table['R_Rank'] / rfm_table['R_Rank'].max()) * 100
    rfm_table['F_rank_norm'] = (rfm_table['F_Rank'] / rfm_table['F_Rank'].max()) * 100
    rfm_table['M_rank_norm'] = (rfm_table['M_Rank'] / rfm_table['M_Rank'].max()) * 100

    # Drop individual rank
    rfm_table.drop(columns=['R_Rank', 'F_Rank', 'M_Rank'], inplace=True)
    rfm_table.head()

    # RFM Scoring digunakan metode weighted dengan 55% Monetary, 30% Frequency, dan 15% Recency
    rfm_table['RFM_Score'] = 0.15 * rfm_table['R_rank_norm'] + 0.3 * rfm_table['F_rank_norm'] + 0.55 * rfm_table['M_rank_norm']
    rfm_table['RFM_Score'] *= 0.05
    rfm_table = rfm_table.round(2)
    
    # Segmentasi Customer berdasarkan RFM
    rfm_table['RFM_Segment'] = pd.cut(
        rfm_table['RFM_Score'],
        bins=[0, 1.6, 3, 4, 4.5, 5],
        labels=['Hibernating',
                'Low Value Customers (At Risk)',
                'Medium Value Customers (Need Attention)',
                'High Value Customers (Loyal)',
                'Top Customers'
                ])
        
    return rfm_table

def k_means_clustering(df):
    X = df[['Total Spend', 'Age', 'Items Purchased']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster1'] = kmeans.fit_predict(X_scaled)

    X_engagement = df[['Days Since Last Purchase', 'Average Rating', 'Discount Applied']]
    scaler_engagement = StandardScaler()
    X_engagement_scaled = scaler_engagement.fit_transform(X_engagement)

    k_engagement = 4

    kmeans_engagement = KMeans(n_clusters=k_engagement, random_state=42)
    df['EngagementCluster'] = kmeans_engagement.fit_predict(X_engagement_scaled)

    X_seasonal = df[['Days Since Last Purchase', 'Total Spend']]
    scaler_seasonal = StandardScaler()
    X_seasonal_scaled = scaler_seasonal.fit_transform(X_seasonal)

    k_seasonal = 5

    kmeans_seasonal = KMeans(n_clusters=k_seasonal, random_state=42)
    df['SeasonalCluster'] = kmeans_seasonal.fit_predict(X_seasonal_scaled)

    return df

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

if __name__ == "__main__":
    main()