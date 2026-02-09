import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ICI Mutual Fund Flows - Growth Dynamics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .analysis-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 10px 20px;
        border-radius: 5px;
        background-color: white;
        border: 1px solid #e5e7eb;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        border-color: #3B82F6 !important;
    }
    .correlation-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with institutional styling
st.markdown('<h1 class="main-header">📈 ICI Mutual Fund Flows - Growth Dynamics Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Visualization of Normalized Growth, Inflow/Outflow Dynamics & Correlation Analysis</p>', unsafe_allow_html=True)

# Sample data generation with realistic patterns
@st.cache_data
def generate_realistic_sample_data(start_date='2007-01-01', end_date=None):
    """Generate realistic sample data with proper correlations and trends"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    n = len(dates)
    np.random.seed(42)
    
    # Base trends with correlations
    time_index = np.arange(n)
    
    # Equity funds - correlated with market cycles
    equity_trend = 0.5 * time_index
    equity_seasonal = 3000 * np.sin(2 * np.pi * time_index / 12)
    equity_random = np.random.normal(0, 4000, n)
    
    # Add market cycles (4-year cycles)
    market_cycle = 8000 * np.sin(2 * np.pi * time_index / 48)
    
    # Add crisis effects
    crisis_2008 = np.exp(-((time_index - 20)**2) / 50) * -25000  # Sep 2008
    covid_2020 = np.exp(-((time_index - 160)**2) / 10) * -35000  # Mar 2020
    
    equity = 10000 + equity_trend + equity_seasonal + equity_random + market_cycle + crisis_2008 + covid_2020
    
    # Bond funds - negatively correlated with equity during crises, positive trend
    bond_trend = 0.3 * time_index
    bond_random = np.random.normal(0, 2000, n)
    # Flight to safety during equity crises
    bond_crisis_2008 = np.exp(-((time_index - 20)**2) / 50) * 15000
    bond_covid_2020 = np.exp(-((time_index - 160)**2) / 10) * 20000
    
    bond = 5000 + bond_trend + bond_random + bond_crisis_2008 + bond_covid_2020
    
    # Money Market - extreme spikes during crises
    mm_trend = 0.1 * time_index
    mm_random = np.random.normal(0, 3000, n)
    mm_crisis_2008 = np.exp(-((time_index - 20)**2) / 30) * 60000
    mm_covid_2020 = np.exp(-((time_index - 160)**2) / 8) * 80000
    
    money_market = 10000 + mm_trend + mm_random + mm_crisis_2008 + mm_covid_2020
    
    # Hybrid funds - mixture of equity and bond
    hybrid = 0.6 * equity + 0.4 * bond + np.random.normal(0, 1000, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Equity': np.round(equity).astype(int),
        'Bond': np.round(bond).astype(int),
        'Hybrid': np.round(hybrid).astype(int),
        'Money Market': np.round(money_market).astype(int)
    })
    
    # Calculate total
    df['Total'] = df[['Equity', 'Bond', 'Hybrid', 'Money Market']].sum(axis=1)
    
    # Calculate cumulative flows
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Cumulative'] = df[category].cumsum()
    
    # Add normalized monthly changes
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Monthly_Change'] = df[category].pct_change() * 100
        df[f'{category}_Normalized'] = 100 * (df[category] - df[category].mean()) / df[category].std()
    
    # Add inflow/outflow indicators
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Inflow'] = df[category].clip(lower=0)
        df[f'{category}_Outflow'] = df[category].clip(upper=0).abs()
    
    return df

# Load data
df = generate_realistic_sample_data()

# Sidebar controls
with st.sidebar:
    st.image("https://www.ici.org/themes/custom/ici/logo.svg", width=200)
    st.markdown("---")
    
    st.header("📊 Growth Analysis Controls")
    
    # Date range
    st.subheader("Time Period")
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        start_date, end_date = st.date_input(
            "Select Analysis Period",
            value=(max_date - timedelta(days=365*5), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    # Categories selection
    st.subheader("Fund Categories")
    categories = st.multiselect(
        "Select categories for analysis",
        ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total'],
        default=['Equity', 'Bond', 'Money Market']
    )
    
    # Normalization options
    st.subheader("Normalization Settings")
    normalize_method = st.selectbox(
        "Normalization Method",
        ["Z-Score", "Min-Max (0-100)", "Percentage of Max", "Log Scale"]
    )
    
    # Display options
    st.subheader("Display Options")
    show_trend_lines = st.checkbox("Show Trend Lines", True)
    show_moving_average = st.checkbox("Show 12-Month Moving Average", True)
    show_correlation = st.checkbox("Show Correlation Matrix", True)
    
    # Analysis depth
    st.subheader("Analysis Depth")
    analysis_level = st.select_slider(
        "Analysis Detail Level",
        options=["Basic", "Intermediate", "Advanced", "Expert"],
        value="Advanced"
    )
    
    st.markdown("---")
    st.markdown("**Note:** All figures in millions of USD")
    st.markdown("Negative values indicate outflows")

# Main content - Growth Dynamics Dashboard
st.markdown('<div class="section-header">🌱 Cumulative Growth Analysis</div>', unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_growth = ((df_filtered['Total_Cumulative'].iloc[-1] - df_filtered['Total_Cumulative'].iloc[0]) 
                   / abs(df_filtered['Total_Cumulative'].iloc[0])) * 100 if len(df_filtered) > 1 else 0
    st.metric(
        "Total Cumulative Growth",
        f"${df_filtered['Total_Cumulative'].iloc[-1]:,.0f}M",
        f"{total_growth:+.1f}%"
    )

with col2:
    equity_growth = ((df_filtered['Equity_Cumulative'].iloc[-1] - df_filtered['Equity_Cumulative'].iloc[0]) 
                    / abs(df_filtered['Equity_Cumulative'].iloc[0])) * 100 if len(df_filtered) > 1 else 0
    st.metric(
        "Equity Growth",
        f"${df_filtered['Equity_Cumulative'].iloc[-1]:,.0f}M",
        f"{equity_growth:+.1f}%"
    )

with col3:
    months_inflow = (df_filtered['Total'] > 0).sum()
    total_months = len(df_filtered)
    inflow_ratio = (months_inflow / total_months) * 100
    st.metric(
        "Inflow Months Ratio",
        f"{months_inflow}/{total_months}",
        f"{inflow_ratio:.1f}%"
    )

with col4:
    volatility = df_filtered['Total'].std() / abs(df_filtered['Total'].mean()) if df_filtered['Total'].mean() != 0 else 0
    st.metric(
        "Flow Volatility",
        f"{volatility:.3f}",
        "Coefficient of Variation"
    )

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Normalized Growth Trends",
    "📈 Inflow/Outflow Dynamics",
    "🔄 Monthly Changes Analysis",
    "🔍 Advanced Statistics"
])

with tab1:
    # Normalized Growth Trends
    st.markdown("### Normalized Cumulative Growth Index (Base = 100)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create normalized growth chart with separate y-axes to prevent overlap
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Equity & Bond Growth', 'Hybrid & Money Market Growth',
                          'Normalized Monthly Flows', 'Relative Performance'],
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[[{'secondary_y': True}, {'secondary_y': True}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Plot 1: Equity vs Bond (top left)
        if 'Equity' in categories and 'Bond' in categories:
            # Normalize to starting point = 100
            equity_norm = 100 * df_filtered['Equity_Cumulative'] / df_filtered['Equity_Cumulative'].iloc[0]
            bond_norm = 100 * df_filtered['Bond_Cumulative'] / df_filtered['Bond_Cumulative'].iloc[0]
            
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=equity_norm,
                          name='Equity', line=dict(color='#1f77b4', width=3),
                          mode='lines'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=bond_norm,
                          name='Bond', line=dict(color='#2ca02c', width=3, dash='dash'),
                          mode='lines'),
                row=1, col=1, secondary_y=True
            )
        
        # Plot 2: Hybrid vs Money Market (top right)
        if 'Hybrid' in categories and 'Money Market' in categories:
            hybrid_norm = 100 * df_filtered['Hybrid_Cumulative'] / df_filtered['Hybrid_Cumulative'].iloc[0]
            mm_norm = 100 * df_filtered['Money Market_Cumulative'] / df_filtered['Money Market_Cumulative'].iloc[0]
            
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=hybrid_norm,
                          name='Hybrid', line=dict(color='#ff7f0e', width=3),
                          mode='lines'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=mm_norm,
                          name='Money Market', line=dict(color='#d62728', width=3, dash='dash'),
                          mode='lines'),
                row=1, col=2, secondary_y=True
            )
        
        # Plot 3: Normalized monthly flows (bottom left)
        if 'Total' in df_filtered.columns:
            # Z-score normalization for monthly flows
            monthly_mean = df_filtered['Total'].mean()
            monthly_std = df_filtered['Total'].std()
            monthly_zscore = (df_filtered['Total'] - monthly_mean) / monthly_std
            
            fig.add_trace(
                go.Bar(x=df_filtered['Date'], y=monthly_zscore,
                      name='Normalized Monthly Flow',
                      marker_color=['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_zscore]),
                row=2, col=1
            )
            
            # Add 12-month moving average
            if show_moving_average:
                ma_12 = monthly_zscore.rolling(window=12).mean()
                fig.add_trace(
                    go.Scatter(x=df_filtered['Date'], y=ma_12,
                              name='12-Month MA',
                              line=dict(color='#2c3e50', width=2)),
                    row=2, col=1
                )
        
        # Plot 4: Relative performance heatmap (bottom right)
        if len(categories) > 1:
            # Calculate monthly returns
            returns_data = []
            for category in categories:
                if f'{category}_Monthly_Change' in df_filtered.columns:
                    returns_data.append(df_filtered[f'{category}_Monthly_Change'].tail(24))  # Last 2 years
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data).T
                returns_df.columns = categories
                
                fig.add_trace(
                    go.Heatmap(z=returns_df.values,
                              x=returns_df.columns,
                              y=returns_df.index,
                              colorscale='RdBu',
                              zmid=0,
                              colorbar=dict(title="Monthly % Change")),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, title_text="Multi-Panel Growth Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Growth Statistics")
        
        growth_stats = []
        for category in categories:
            if f'{category}_Cumulative' in df_filtered.columns:
                start_val = df_filtered[f'{category}_Cumulative'].iloc[0]
                end_val = df_filtered[f'{category}_Cumulative'].iloc[-1]
                total_growth = end_val - start_val
                growth_pct = (total_growth / abs(start_val)) * 100 if start_val != 0 else 0
                
                # Calculate CAGR
                years = len(df_filtered) / 12
                if years > 0 and start_val != 0:
                    cagr = ((end_val / abs(start_val)) ** (1/years) - 1) * 100
                else:
                    cagr = 0
                
                growth_stats.append({
                    'Category': category,
                    'Total Growth': f"${total_growth:,.0f}M",
                    'Growth %': f"{growth_pct:+.1f}%",
                    'CAGR': f"{cagr:+.1f}%"
                })
        
        if growth_stats:
            stats_df = pd.DataFrame(growth_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab2:
    # Inflow/Outflow Dynamics
    st.markdown("### Separate Inflow and Outflow Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monthly Inflows by Category")
        
        inflow_data = []
        for category in categories:
            if f'{category}_Inflow' in df_filtered.columns:
                inflow_data.append(go.Bar(
                    name=category,
                    x=df_filtered['Date'],
                    y=df_filtered[f'{category}_Inflow'],
                    opacity=0.7
                ))
        
        if inflow_data:
            fig_inflow = go.Figure(data=inflow_data)
            fig_inflow.update_layout(
                barmode='stack',
                title="Total Monthly Inflows",
                xaxis_title="Date",
                yaxis_title="Millions USD (Inflows)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_inflow, use_container_width=True)
        
        # Inflow statistics
        st.markdown("**Inflow Statistics**")
        inflow_stats = []
        for category in categories:
            if f'{category}_Inflow' in df_filtered.columns:
                total_in = df_filtered[f'{category}_Inflow'].sum()
                avg_in = df_filtered[f'{category}_Inflow'].mean()
                max_in = df_filtered[f'{category}_Inflow'].max()
                
                inflow_stats.append({
                    'Category': category,
                    'Total Inflow': f"${total_in:,.0f}M",
                    'Average': f"${avg_in:,.0f}M",
                    'Peak': f"${max_in:,.0f}M"
                })
        
        if inflow_stats:
            st.dataframe(pd.DataFrame(inflow_stats), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Monthly Outflows by Category")
        
        outflow_data = []
        for category in categories:
            if f'{category}_Outflow' in df_filtered.columns:
                outflow_data.append(go.Bar(
                    name=category,
                    x=df_filtered['Date'],
                    y=df_filtered[f'{category}_Outflow'],
                    opacity=0.7
                ))
        
        if outflow_data:
            fig_outflow = go.Figure(data=outflow_data)
            fig_outflow.update_layout(
                barmode='stack',
                title="Total Monthly Outflows",
                xaxis_title="Date",
                yaxis_title="Millions USD (Outflows)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_outflow, use_container_width=True)
        
        # Outflow statistics
        st.markdown("**Outflow Statistics**")
        outflow_stats = []
        for category in categories:
            if f'{category}_Outflow' in df_filtered.columns:
                total_out = df_filtered[f'{category}_Outflow'].sum()
                avg_out = df_filtered[f'{category}_Outflow'].mean()
                max_out = df_filtered[f'{category}_Outflow'].max()
                
                outflow_stats.append({
                    'Category': category,
                    'Total Outflow': f"${total_out:,.0f}M",
                    'Average': f"${avg_out:,.0f}M",
                    'Peak': f"${max_out:,.0f}M"
                })
        
        if outflow_stats:
            st.dataframe(pd.DataFrame(outflow_stats), use_container_width=True, hide_index=True)
    
    # Net flow analysis
    st.markdown("---")
    st.markdown("#### Net Flow Analysis")
    
    net_flow_fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=['Net Flow by Category', 'Inflow-Outflow Ratio',
                                               'Cumulative Net Flow', 'Flow Direction Over Time'])
    
    # Net flow by category
    for category in categories:
        if category in df_filtered.columns:
            net_flow_fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered[category],
                          name=category, mode='lines'),
                row=1, col=1
            )
    
    # Inflow-Outflow ratio
    for category in categories:
        if f'{category}_Inflow' in df_filtered.columns and f'{category}_Outflow' in df_filtered.columns:
            ratio = df_filtered[f'{category}_Inflow'] / (df_filtered[f'{category}_Outflow'] + 1)
            net_flow_fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=ratio,
                          name=f"{category} Ratio", mode='lines'),
                row=1, col=2
            )
    
    # Cumulative net flow
    for category in categories:
        if f'{category}_Cumulative' in df_filtered.columns:
            net_flow_fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered[f'{category}_Cumulative'],
                          name=f"{category} Cumulative", mode='lines'),
                row=2, col=1
            )
    
    # Flow direction (positive/negative months)
    direction_data = []
    for category in categories:
        if category in df_filtered.columns:
            positive_months = (df_filtered[category] > 0).sum()
            negative_months = (df_filtered[category] < 0).sum()
            direction_data.append(go.Bar(
                name=category,
                x=['Inflow Months', 'Outflow Months'],
                y=[positive_months, negative_months]
            ))
    
    for i, trace in enumerate(direction_data):
        net_flow_fig.add_trace(trace, row=2, col=2)
    
    net_flow_fig.update_layout(height=700, showlegend=True)
    net_flow_fig.update_yaxes(title_text="Millions USD", row=1, col=1)
    net_flow_fig.update_yaxes(title_text="Ratio", row=1, col=2)
    net_flow_fig.update_yaxes(title_text="Cumulative USD", row=2, col=1)
    net_flow_fig.update_yaxes(title_text="Number of Months", row=2, col=2)
    
    st.plotly_chart(net_flow_fig, use_container_width=True)

with tab3:
    # Monthly Changes Analysis
    st.markdown("### Monthly Percentage Changes & Normalized Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monthly Percentage Changes")
        
        change_fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, category in enumerate(categories):
            if f'{category}_Monthly_Change' in df_filtered.columns:
                change_fig.add_trace(go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered[f'{category}_Monthly_Change'],
                    name=category,
                    mode='lines+markers',
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=4),
                    hovertemplate='%{x|%b %Y}<br>' +
                                f'{category}: %{{y:.1f}}%'
                ))
        
        change_fig.update_layout(
            title="Monthly Percentage Changes by Category",
            xaxis_title="Date",
            yaxis_title="Monthly % Change",
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add zero line
        change_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(change_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Normalized Monthly Flows (Z-Score)")
        
        norm_fig = go.Figure()
        
        for i, category in enumerate(categories):
            if f'{category}_Normalized' in df_filtered.columns:
                norm_fig.add_trace(go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered[f'{category}_Normalized'],
                    name=category,
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate='%{x|%b %Y}<br>' +
                                f'{category}: %{{y:.2f}}σ'
                ))
        
        norm_fig.update_layout(
            title="Normalized Monthly Flows (Z-Score)",
            xaxis_title="Date",
            yaxis_title="Standard Deviations from Mean",
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add zero line
        norm_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(norm_fig, use_container_width=True)
    
    # Rolling statistics
    st.markdown("---")
    st.markdown("#### Rolling Statistics Analysis")
    
    rolling_cols = st.columns(3)
    
    with rolling_cols[0]:
        window = st.slider("Rolling Window (months)", 3, 24, 12)
    
    with rolling_cols[1]:
        stat_type = st.selectbox("Statistic", ["Mean", "Std Dev", "Max", "Min", "Volatility"])
    
    with rolling_cols[2]:
        category_for_rolling = st.selectbox("Category for Rolling Stats", categories)
    
    if category_for_rolling in df_filtered.columns:
        rolling_stats_fig = make_subplots(rows=2, cols=2,
                                         subplot_titles=['Rolling Mean', 'Rolling Standard Deviation',
                                                        'Rolling Max/Min', 'Rolling Volatility'])
        
        # Rolling mean
        rolling_mean = df_filtered[category_for_rolling].rolling(window=window).mean()
        rolling_stats_fig.add_trace(
            go.Scatter(x=df_filtered['Date'], y=rolling_mean,
                      name=f'{window}-Month MA', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Rolling std dev
        rolling_std = df_filtered[category_for_rolling].rolling(window=window).std()
        rolling_stats_fig.add_trace(
            go.Scatter(x=df_filtered['Date'], y=rolling_std,
                      name=f'{window}-Month Std Dev', line=dict(color='red')),
            row=1, col=2
        )
        
        # Rolling max/min
        rolling_max = df_filtered[category_for_rolling].rolling(window=window).max()
        rolling_min = df_filtered[category_for_rolling].rolling(window=window).min()
        rolling_stats_fig.add_trace(
            go.Scatter(x=df_filtered['Date'], y=rolling_max,
                      name='Rolling Max', line=dict(color='green')),
            row=2, col=1
        )
        rolling_stats_fig.add_trace(
            go.Scatter(x=df_filtered['Date'], y=rolling_min,
                      name='Rolling Min', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Rolling volatility (std/mean)
        rolling_vol = rolling_std / rolling_mean.abs()
        rolling_stats_fig.add_trace(
            go.Scatter(x=df_filtered['Date'], y=rolling_vol,
                      name='Rolling Volatility', line=dict(color='purple')),
            row=2, col=2
        )
        
        rolling_stats_fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(rolling_stats_fig, use_container_width=True)

with tab4:
    # Advanced Statistics
    st.markdown("### Advanced Statistical Analysis")
    
    # Correlation Analysis
    if show_correlation and len(categories) > 1:
        st.markdown("#### Correlation Matrix")
        
        # Prepare correlation data
        corr_data = []
        for category in categories:
            if category in df_filtered.columns:
                corr_data.append(df_filtered[category])
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data).T
            corr_df.columns = categories
            correlation_matrix = corr_df.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Correlation Matrix Between Categories",
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.markdown("**Correlation Insights:**")
            insights = []
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = correlation_matrix.loc[cat1, cat2]
                        if abs(corr_value) > 0.7:
                            strength = "Strong" if abs(corr_value) > 0.8 else "Moderate"
                            direction = "positive" if corr_value > 0 else "negative"
                            insights.append(f"**{cat1}** and **{cat2}** have a {strength} {direction} correlation ({corr_value:.2f})")
            
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("No strong correlations found between selected categories.")
    
    # Distribution Analysis
    st.markdown("---")
    st.markdown("#### Distribution Analysis")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Histogram of flows
        hist_fig = go.Figure()
        
        for category in categories:
            if category in df_filtered.columns:
                hist_fig.add_trace(go.Histogram(
                    x=df_filtered[category],
                    name=category,
                    opacity=0.6,
                    nbinsx=30
                ))
        
        hist_fig.update_layout(
            title="Distribution of Monthly Flows",
            xaxis_title="Monthly Flow (Millions USD)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with dist_col2:
        # Box plot
        box_data = []
        for category in categories:
            if category in df_filtered.columns:
                box_data.append(go.Box(
                    y=df_filtered[category],
                    name=category,
                    boxpoints='outliers'
                ))
        
        if box_data:
            box_fig = go.Figure(data=box_data)
            box_fig.update_layout(
                title="Box Plot of Monthly Flows",
                yaxis_title="Monthly Flow (Millions USD)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(box_fig, use_container_width=True)
    
    # Time Series Decomposition
    if analysis_level in ["Advanced", "Expert"]:
        st.markdown("---")
        st.markdown("#### Time Series Decomposition")
        
        decompose_category = st.selectbox("Select category for decomposition", categories)
        
        if decompose_category in df_filtered.columns:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Ensure we have a regular time series
            ts_data = df_filtered.set_index('Date')[decompose_category]
            ts_data = ts_data.asfreq('MS').fillna(method='ffill')
            
            try:
                decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                
                decomp_fig = make_subplots(rows=4, cols=1,
                                          subplot_titles=['Original Series', 'Trend',
                                                         'Seasonality', 'Residuals'])
                
                decomp_fig.add_trace(
                    go.Scatter(x=ts_data.index, y=ts_data.values,
                              name='Original', line=dict(color='blue')),
                    row=1, col=1
                )
                
                decomp_fig.add_trace(
                    go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                              name='Trend', line=dict(color='red')),
                    row=2, col=1
                )
                
                decomp_fig.add_trace(
                    go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                              name='Seasonal', line=dict(color='green')),
                    row=3, col=1
                )
                
                decomp_fig.add_trace(
                    go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                              name='Residual', line=dict(color='orange')),
                    row=4, col=1
                )
                
                decomp_fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(decomp_fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not perform time series decomposition: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Investment Company Institute (ICI) Mutual Fund Flows Analysis | Data Source: ICI Statistics</p>
    <p>All figures in millions of USD | Negative values indicate net outflows</p>
    <p>Last Updated: Sample Data Generated</p>
</div>
""", unsafe_allow_html=True)
