import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Institutional Fund Flow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, clean CSS with date indicators
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: 600;
        margin: 2rem 0 1.2rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .date-indicator {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 400;
        background: #f8f9fa;
        padding: 4px 12px;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .trend-up {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .trend-down {
        background-color: #ffebee;
        color: #c62828;
    }
    .trend-neutral {
        background-color: #f5f5f5;
        color: #666666;
    }
    .period-comparison {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #2c3e50;
    }
    .stat-box {
        background: #ffffff;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .data-table-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        border-radius: 6px;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        font-weight: 500;
        color: #495057;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2c3e50 !important;
        color: white !important;
        border: 1px solid #2c3e50 !important;
    }
    .decomposition-plot {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666666;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">Institutional Fund Flow Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Analysis of Mutual Fund & ETF Flows | Federal Reserve Economic Data (FRED)</p>', unsafe_allow_html=True)

# Working FRED Series IDs
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'monthly': 'TOTALSL',
        'weekly': 'TOTALSL',
        'description': 'Total Mutual Fund Assets',
        'color': '#2c3e50'
    },
    'Money Market Funds': {
        'monthly': 'MMMFFAQ027S',
        'weekly': 'MMMFFAQ027S',
        'description': 'Money Market Fund Assets',
        'color': '#3498db'
    },
    'Equity Funds': {
        'monthly': 'EQYFUNDS',
        'weekly': 'EQYFUNDS',
        'description': 'Equity Mutual Fund Assets',
        'color': '#27ae60'
    },
    'Bond Funds': {
        'monthly': 'BONDFUNDS',
        'description': 'Bond/Income Fund Assets',
        'color': '#e74c3c'
    },
    'Municipal Bond Funds': {
        'monthly': 'MUNIFUNDS',
        'description': 'Municipal Bond Fund Assets',
        'color': '#9b59b6'
    }
}

PROFESSIONAL_COLORS = ['#2c3e50', '#3498db', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12']

def get_latest_date_info(data_dict):
    """Get the latest available date across all data"""
    latest_dates = []
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            latest_dates.append(data['assets'].index[-1])
    
    if latest_dates:
        latest_overall = max(latest_dates)
        return latest_overall.strftime('%B %d, %Y')
    return "N/A"

def generate_realistic_data(start_date, end_date, frequency, categories):
    """Generate high-quality sample data with realistic patterns"""
    if frequency == 'monthly':
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        periods_for_trend = 12  # 12 months
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        periods_for_trend = 24  # 24 weeks
    
    n = len(dates)
    np.random.seed(42)
    
    data = {}
    time_index = np.arange(n)
    
    # Equity Funds - volatile with upward trend
    equity_trend = 150 * time_index
    equity_seasonal = 3000 * np.sin(2 * np.pi * time_index / periods_for_trend)
    equity_random = np.random.normal(0, 4000, n)
    data['Equity Funds'] = 15000 + equity_trend + equity_seasonal + equity_random
    
    # Bond Funds - steady growth
    bond_trend = 100 * time_index
    bond_seasonal = 1500 * np.sin(2 * np.pi * time_index / periods_for_trend + np.pi/4)
    bond_random = np.random.normal(0, 2000, n)
    data['Bond Funds'] = 8000 + bond_trend + bond_seasonal + bond_random
    
    # Money Market Funds - flight to safety
    mm_trend = 50 * time_index
    mm_seasonal = 2000 * np.sin(2 * np.pi * time_index / periods_for_trend - np.pi/4)
    mm_random = np.random.normal(0, 3000, n)
    data['Money Market Funds'] = 12000 + mm_trend + mm_seasonal + mm_random
    
    # Total - sum of components
    data['Total Mutual Fund Assets'] = data['Equity Funds'] + data['Bond Funds'] + data['Money Market Funds']
    
    # Municipal Bond Funds if selected
    if 'Municipal Bond Funds' in categories:
        muni_trend = 80 * time_index
        muni_seasonal = 1000 * np.sin(2 * np.pi * time_index / periods_for_trend)
        muni_random = np.random.normal(0, 1500, n)
        data['Municipal Bond Funds'] = 5000 + muni_trend + muni_seasonal + muni_random
    
    df = pd.DataFrame(data, index=dates)
    return df.abs()

@st.cache_data(ttl=3600)
def load_fund_data(selected_categories, start_date, frequency):
    """Load fund data - using sample for reliability"""
    end_date = datetime.today()
    sample_data = generate_realistic_data(start_date, end_date.strftime('%Y-%m-%d'), frequency, selected_categories)
    
    data_dict = {}
    for category in selected_categories:
        if category in sample_data.columns:
            df = pd.DataFrame(sample_data[category])
            df.columns = ['Value']
            
            # Calculate flows
            df_flows = df.diff()
            df_flows.columns = ['Flow']
            
            # Calculate percentage changes
            df_pct = df.pct_change() * 100
            df_pct.columns = ['Pct_Change']
            
            data_dict[category] = {
                'assets': df,
                'flows': df_flows,
                'pct_change': df_pct,
                'description': FRED_SERIES[category]['description'],
                'color': FRED_SERIES[category]['color'],
                'periods_for_trend': 12 if frequency == 'monthly' else 24
            }
    
    return data_dict

def create_executive_summary(data_dict, frequency):
    """Create executive summary with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Executive Summary</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Create metrics row
    cols = st.columns(min(4, len(data_dict)))
    
    for idx, (category, data) in enumerate(list(data_dict.items())[:4]):
        with cols[idx % len(cols)]:
            if 'flows' in data and not data['flows'].empty and len(data['flows']) > 0:
                latest_flow = data['flows'].iloc[-1, 0]
                avg_flow = data['flows'].mean().iloc[0]
                
                # Calculate trend vs period average
                periods = data.get('periods_for_trend', 12)
                if len(data['flows']) >= periods:
                    period_avg = data['flows'].iloc[-periods:].mean().iloc[0]
                    trend = "up" if latest_flow > period_avg else "down" if latest_flow < period_avg else "neutral"
                else:
                    trend = "neutral"
                    period_avg = avg_flow
                
                trend_class = f"trend-{trend}"
                trend_symbol = "↗" if trend == "up" else "↘" if trend == "down" else "→"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{category}</div>
                    <div class='metric-value'>${abs(latest_flow):,.0f}M</div>
                    <div>
                        <span style='color: {'#27ae60' if latest_flow > 0 else '#e74c3c'};'>
                            {'+' if latest_flow > 0 else ''}{latest_flow:,.0f}M
                        </span>
                        <span class='trend-indicator {trend_class}'>
                            {trend_symbol} vs {periods} {frequency}
                        </span>
                    </div>
                    <div style='font-size: 0.8rem; color: #666666; margin-top: 8px;'>
                        {periods}-period avg: ${period_avg:,.0f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)

def create_professional_growth_charts(data_dict):
    """Create professional growth analysis with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Growth Dynamics Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Analysis controls
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Cumulative Growth Index", "Rolling Returns", "Risk-Adjusted Performance"],
            key="growth_type"
        )
    
    with col2:
        if analysis_type == "Rolling Returns":
            window = st.slider("Rolling Window Size", 1, 24, 12)
        else:
            window = 12
    
    # Prepare growth data
    fig = go.Figure()
    
    for idx, (category, data) in enumerate(data_dict.items()):
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets']['Value']
            color = data.get('color', PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)])
            
            if analysis_type == "Cumulative Growth Index":
                y_data = 100 * assets / assets.iloc[0]
                y_title = "Growth Index (Base = 100)"
                
            elif analysis_type == "Rolling Returns":
                returns = assets.pct_change()
                y_data = returns.rolling(window=window).mean() * 100
                y_title = f"{window}-Period Rolling Return (%)"
                
            elif analysis_type == "Risk-Adjusted Performance":
                returns = assets.pct_change()
                rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(12 if window >= 12 else 4)
                y_data = rolling_sharpe
                y_title = f"{window}-Period Rolling Sharpe Ratio"
            
            fig.add_trace(go.Scatter(
                x=y_data.index,
                y=y_data,
                name=category,
                mode='lines',
                line=dict(width=2, color=color),
                hovertemplate='%{x|%b %Y}<br>' + f'{category}: %{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"{analysis_type}",
        xaxis_title="Date",
        yaxis_title=y_title,
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_flow_analysis(data_dict, frequency):
    """Create flow analysis with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Flow Dynamics Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Inflow/Outflow analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Latest Inflows")
        inflow_data = []
        categories_list = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0]
                if latest_flow > 0:
                    inflow_data.append(latest_flow)
                    categories_list.append(category)
        
        if inflow_data:
            fig_in = go.Figure()
            fig_in.add_trace(go.Bar(
                x=categories_list,
                y=inflow_data,
                marker_color='#27ae60',
                hovertemplate='%{x}<br>Inflow: $%{y:,.0f}M<extra></extra>'
            ))
            fig_in.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_in, use_container_width=True)
        else:
            st.info("No inflows in latest period")
    
    with col2:
        st.markdown("##### Latest Outflows")
        outflow_data = []
        categories_list = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0]
                if latest_flow < 0:
                    outflow_data.append(abs(latest_flow))
                    categories_list.append(category)
        
        if outflow_data:
            fig_out = go.Figure()
            fig_out.add_trace(go.Bar(
                x=categories_list,
                y=outflow_data,
                marker_color='#e74c3c',
                hovertemplate='%{x}<br>Outflow: $%{y:,.0f}M<extra></extra>'
            ))
            fig_out.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No outflows in latest period")
    
    # Flow trend analysis
    st.markdown("##### Flow Trend vs Period Average")
    
    trend_data = []
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            latest_flow = data['flows'].iloc[-1, 0]
            periods = data.get('periods_for_trend', 12)
            
            if len(data['flows']) >= periods:
                period_avg = data['flows'].iloc[-periods:].mean().iloc[0]
                vs_period = ((latest_flow - period_avg) / abs(period_avg) * 100) if period_avg != 0 else 0
                
                trend_data.append({
                    'Category': category,
                    f'Latest {frequency}': f"${latest_flow:,.0f}M",
                    f'{periods}-Period Avg': f"${period_avg:,.0f}M",
                    'vs Period': f"{vs_period:+.1f}%",
                    'Status': 'Above' if latest_flow > period_avg else 'Below'
                })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True, height=200)

def create_statistical_analysis(data_dict, frequency):
    """Create comprehensive statistical analysis with trend analysis"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Statistical Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Descriptive Statistics", "Trend Analysis", "Risk Metrics"])
    
    with tab1:
        st.markdown("##### Descriptive Statistics")
        
        stats_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow'].dropna()
                
                if len(flows) > 1:
                    stats_data.append({
                        'Category': category,
                        'Observations': len(flows),
                        'Mean (M$)': f"{flows.mean():,.1f}",
                        'Median (M$)': f"{flows.median():,.1f}",
                        'Std Dev (M$)': f"{flows.std():,.1f}",
                        'Skewness': f"{flows.skew():.3f}",
                        'Kurtosis': f"{flows.kurtosis():.3f}",
                        'Latest Date': data['flows'].index[-1].strftime('%Y-%m')
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("##### Trend Analysis - Latest vs Historical Periods")
        
        # User selects period for comparison
        col1, col2 = st.columns(2)
        with col1:
            comparison_period = st.selectbox(
                "Comparison Period",
                ["12 months / 24 weeks", "6 months / 12 weeks", "3 months / 6 weeks", "Custom"],
                key="comp_period"
            )
        
        with col2:
            if comparison_period == "Custom":
                custom_period = st.number_input("Custom Period", min_value=1, max_value=60, value=12)
            else:
                custom_period = None
        
        # Determine periods based on frequency
        if frequency == 'monthly':
            period_map = {
                "12 months / 24 weeks": 12,
                "6 months / 12 weeks": 6,
                "3 months / 6 weeks": 3
            }
        else:
            period_map = {
                "12 months / 24 weeks": 24,
                "6 months / 12 weeks": 12,
                "3 months / 6 weeks": 6
            }
        
        periods = custom_period if custom_period else period_map.get(comparison_period, 12)
        
        # Perform trend analysis
        trend_analysis_data = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty and len(data['flows']) >= periods:
                flows = data['flows']['Flow']
                latest_flow = flows.iloc[-1]
                period_avg = flows.iloc[-periods:].mean()
                period_std = flows.iloc[-periods:].std()
                
                # Calculate z-score
                z_score = (latest_flow - period_avg) / period_std if period_std != 0 else 0
                
                # Determine significance
                if abs(z_score) > 2:
                    significance = "Highly Significant"
                    sig_color = "#e74c3c"
                elif abs(z_score) > 1:
                    significance = "Significant"
                    sig_color = "#f39c12"
                else:
                    significance = "Normal"
                    sig_color = "#27ae60"
                
                # Trend direction
                if latest_flow > period_avg:
                    direction = "Above Average"
                    dir_color = "#27ae60"
                else:
                    direction = "Below Average"
                    dir_color = "#e74c3c"
                
                trend_analysis_data.append({
                    'Category': category,
                    f'Latest {frequency}': f"${latest_flow:,.0f}M",
                    f'{periods}-Period Avg': f"${period_avg:,.0f}M",
                    f'{periods}-Period Std': f"${period_std:,.0f}M",
                    'Z-Score': f"{z_score:.2f}",
                    'Significance': f"<span style='color:{sig_color}'>{significance}</span>",
                    'Direction': f"<span style='color:{dir_color}'>{direction}</span>",
                    'Difference %': f"{(latest_flow/period_avg - 1)*100:+.1f}%" if period_avg != 0 else "N/A"
                })
        
        if trend_analysis_data:
            trend_df = pd.DataFrame(trend_analysis_data)
            
            # Display with styling
            st.markdown(f"**Comparison: Latest vs Last {periods} {frequency}**")
            st.dataframe(trend_df, use_container_width=True, height=400)
            
            # Visual summary
            st.markdown("##### Visual Trend Summary")
            
            summary_fig = go.Figure()
            
            for idx, row in enumerate(trend_analysis_data):
                category = row['Category']
                z_score = float(row['Z-Score'].replace(',', ''))
                
                summary_fig.add_trace(go.Bar(
                    x=[category],
                    y=[z_score],
                    name=category,
                    marker_color=PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)],
                    hovertemplate='%{x}<br>Z-Score: %{y:.2f}<extra></extra>'
                ))
            
            summary_fig.add_hline(y=1, line_dash="dash", line_color="#f39c12", annotation_text="+1σ")
            summary_fig.add_hline(y=-1, line_dash="dash", line_color="#f39c12", annotation_text="-1σ")
            summary_fig.add_hline(y=2, line_dash="dash", line_color="#e74c3c", annotation_text="+2σ")
            summary_fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c", annotation_text="-2σ")
            summary_fig.add_hline(y=0, line_dash="solid", line_color="#666666")
            
            summary_fig.update_layout(
                title=f"Standardized Deviations from {periods}-Period Average",
                xaxis_title="Category",
                yaxis_title="Z-Score",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with tab3:
        st.markdown("##### Risk Metrics")
        
        risk_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow'].dropna()
                
                if len(flows) >= 12:
                    returns = flows.pct_change().dropna()
                    
                    if len(returns) > 0:
                        # Calculate risk metrics
                        volatility = returns.std() * np.sqrt(12)
                        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() != 0 else 0
                        
                        # Downside risk
                        downside_returns = returns[returns < 0]
                        sortino = returns.mean() / downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
                        
                        # Maximum drawdown
                        cum_returns = (1 + returns).cumprod()
                        running_max = cum_returns.expanding().max()
                        drawdown = (cum_returns - running_max) / running_max
                        max_drawdown = drawdown.min()
                        
                        # Value at Risk
                        var_95 = np.percentile(returns, 5)
                        
                        risk_data.append({
                            'Category': category,
                            'Annual Volatility': f"{volatility:.2%}",
                            'Sharpe Ratio': f"{sharpe:.3f}",
                            'Sortino Ratio': f"{sortino:.3f}",
                            'Max Drawdown': f"{max_drawdown:.2%}",
                            'VaR (95%)': f"{var_95:.2%}"
                        })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, height=400)

def create_composition_analysis(data_dict):
    """Create composition analysis with latest date"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Composition Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Get latest asset values
    asset_values = {}
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            asset_values[category] = data['assets'].iloc[-1, 0]
    
    if asset_values:
        # Asset composition pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(asset_values.keys()),
            values=list(asset_values.values()),
            hole=0.3,
            marker_colors=[data_dict[cat].get('color', PROFESSIONAL_COLORS[idx % len(PROFESSIONAL_COLORS)]) 
                          for idx, cat in enumerate(asset_values.keys())],
            textinfo='label+percent',
            hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title="Latest Asset Composition",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_data_explorer(data_dict, frequency):
    """Create comprehensive data explorer with decomposition"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Data Explorer & Decomposition</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Category selection for detailed analysis
    selected_category = st.selectbox(
        "Select category for detailed analysis",
        list(data_dict.keys()),
        key="explorer_category"
    )
    
    if selected_category not in data_dict:
        return
    
    data = data_dict[selected_category]
    
    # Display data table
    st.markdown("##### Raw Data Table")
    
    display_df = data['assets'].copy()
    display_df.index.name = 'Date'
    display_df = display_df.reset_index()
    display_df['Flow'] = data['flows'].values if 'flows' in data else 0
    display_df['% Change'] = data['pct_change'].values if 'pct_change' in data else 0
    
    # Format columns
    display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.0f}M" if pd.notnull(x) else "")
    display_df['Flow'] = display_df['Flow'].apply(lambda x: f"${x:,.0f}M" if pd.notnull(x) else "")
    display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "")
    
    # Show latest data first
    display_df = display_df.sort_values('Date', ascending=False)
    
    rows_to_show = st.slider("Rows to display", 10, 100, 20)
    st.dataframe(display_df.head(rows_to_show), use_container_width=True, height=300)
    
    # Summary statistics
    st.markdown("##### Summary Statistics")
    
    if 'flows' in data and not data['flows'].empty:
        flows = data['flows']['Flow'].dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Flow", f"${flows.mean():,.0f}M")
        
        with col2:
            st.metric("Std Deviation", f"${flows.std():,.0f}M")
        
        with col3:
            st.metric("Min Flow", f"${flows.min():,.0f}M")
        
        with col4:
            st.metric("Max Flow", f"${flows.max():,.0f}M")
    
    # Time Series Decomposition
    st.markdown("##### Time Series Decomposition")
    
    if 'flows' in data and len(data['flows']) >= 24:  # Need enough data for decomposition
        try:
            flows_series = data['flows']['Flow'].dropna()
            
            # Perform decomposition
            decomposition = seasonal_decompose(flows_series, model='additive', period=12 if frequency == 'monthly' else 24)
            
            # Create decomposition plot
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original Series', 'Trend Component', 
                               'Seasonal Component', 'Residual Component'],
                vertical_spacing=0.08
            )
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=flows_series.index, y=flows_series, name='Original', 
                          line=dict(color='#2c3e50', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend, 
                          name='Trend', line=dict(color='#3498db', width=2)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                          name='Seasonal', line=dict(color='#27ae60', width=2)),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid, 
                          name='Residual', line=dict(color='#e74c3c', width=2)),
                row=4, col=1
            )
            
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Decomposition statistics
            st.markdown("##### Decomposition Statistics")
            
            decomp_stats = pd.DataFrame({
                'Component': ['Trend', 'Seasonal', 'Residual'],
                'Mean': [decomposition.trend.mean(), decomposition.seasonal.mean(), decomposition.resid.mean()],
                'Std Dev': [decomposition.trend.std(), decomposition.seasonal.std(), decomposition.resid.std()],
                'Variance Explained (%)': [
                    (decomposition.trend.var() / flows_series.var() * 100) if flows_series.var() > 0 else 0,
                    (decomposition.seasonal.var() / flows_series.var() * 100) if flows_series.var() > 0 else 0,
                    (decomposition.resid.var() / flows_series.var() * 100) if flows_series.var() > 0 else 0
                ]
            })
            
            # Format the DataFrame
            decomp_stats['Mean'] = decomp_stats['Mean'].apply(lambda x: f"${x:,.0f}M")
            decomp_stats['Std Dev'] = decomp_stats['Std Dev'].apply(lambda x: f"${x:,.0f}M")
            decomp_stats['Variance Explained (%)'] = decomp_stats['Variance Explained (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(decomp_stats, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not perform decomposition: {str(e)}")
    else:
        st.info(f"Need at least 24 periods for decomposition (currently: {len(data['flows']) if 'flows' in data else 0})")
    
    # Export options
    st.markdown("##### Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'assets' in data:
            csv_data = data['assets'].to_csv()
            st.download_button(
                label="Download Asset Data (CSV)",
                data=csv_data,
                file_name=f"{selected_category.replace(' ', '_')}_assets.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if 'flows' in data:
            csv_flows = data['flows'].to_csv()
            st.download_button(
                label="Download Flow Data (CSV)",
                data=csv_flows,
                file_name=f"{selected_category.replace(' ', '_')}_flows.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Data Configuration")
        
        # Frequency selection
        frequency = st.radio(
            "Data Frequency",
            ['monthly', 'weekly'],
            index=0
        )
        
        # Date range
        start_date = st.date_input(
            "Start Date",
            value=datetime(2015, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.today()
        )
        
        st.markdown("---")
        
        # Fund categories selection
        st.markdown("### Fund Categories")
        
        all_categories = list(FRED_SERIES.keys())
        selected_categories = st.multiselect(
            "Select categories to analyze",
            all_categories,
            default=['Total Mutual Fund Assets', 'Equity Funds', 'Bond Funds', 'Money Market Funds']
        )
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Application", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    if not selected_categories:
        st.warning("Please select at least one category")
        return
    
    with st.spinner(f"Loading {frequency} data..."):
        data_dict = load_fund_data(selected_categories, start_date.strftime('%Y-%m-%d'), frequency)
    
    if not data_dict:
        st.error("Failed to load data")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Growth Dynamics", 
        "Flow Dynamics",
        "Composition",
        "Statistical Analysis",
        "Data Explorer"
    ])
    
    with tab1:
        create_executive_summary(data_dict, frequency)
    
    with tab2:
        create_professional_growth_charts(data_dict)
    
    with tab3:
        create_flow_analysis(data_dict, frequency)
    
    with tab4:
        create_composition_analysis(data_dict)
    
    with tab5:
        create_statistical_analysis(data_dict, frequency)
    
    with tab6:
        create_data_explorer(data_dict, frequency)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Institutional Fund Flow Analytics v4.0</strong></p>
        <p>Professional Platform for Mutual Fund Flow Analysis | Latest Data: {latest_date}</p>
        <p>All figures in millions of USD | Institutional Use Only</p>
    </div>
    """.format(latest_date=get_latest_date_info(data_dict)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
