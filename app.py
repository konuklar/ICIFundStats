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

# Professional, clean CSS with enhanced styling
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
    .flow-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 4px;
    }
    .flow-in {
        background-color: rgba(39, 174, 96, 0.1);
        color: #27ae60;
        border: 1px solid rgba(39, 174, 96, 0.3);
    }
    .flow-out {
        background-color: rgba(231, 76, 60, 0.1);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .chart-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .analysis-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2c3e50;
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
st.markdown('<p class="sub-header">Professional Analysis of Mutual Fund & ETF Flows | Advanced Flow Dynamics</p>', unsafe_allow_html=True)

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
            window = st.slider("Rolling Window Size", 1, 24, 12, key="growth_window")
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

def create_enhanced_flow_analysis(data_dict, frequency):
    """Create enhanced institutional flow analysis with trend and Bollinger bands"""
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
    
    # Create overview metrics
    st.markdown("### Flow Overview Dashboard")
    
    # Calculate summary metrics
    total_inflows = 0
    total_outflows = 0
    inflow_categories = []
    outflow_categories = []
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            latest_flow = data['flows'].iloc[-1, 0]
            if latest_flow > 0:
                total_inflows += latest_flow
                inflow_categories.append(category)
            else:
                total_outflows += abs(latest_flow)
                outflow_categories.append(category)
    
    # Display flow overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Inflows</div>
            <div class='metric-value'>${total_inflows:,.0f}M</div>
            <div class='flow-indicator flow-in'>{len(inflow_categories)} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Outflows</div>
            <div class='metric-value'>${total_outflows:,.0f}M</div>
            <div class='flow-indicator flow-out'>{len(outflow_categories)} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        net_flow = total_inflows - total_outflows
        net_color = "#27ae60" if net_flow > 0 else "#e74c3c"
        net_icon = "↗️" if net_flow > 0 else "↘️"
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Net Flow</div>
            <div class='metric-value' style='color: {net_color}'>{net_icon} ${abs(net_flow):,.0f}M</div>
            <div style='font-size: 0.85rem; color: #666666;'>
                {'Positive' if net_flow > 0 else 'Negative'} Net Flow
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        flow_ratio = total_inflows / total_outflows if total_outflows > 0 else float('inf')
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Inflow/Outflow Ratio</div>
            <div class='metric-value'>{flow_ratio:.2f}:1</div>
            <div style='font-size: 0.85rem; color: #666666;'>
                {'Inflow Dominant' if flow_ratio > 1 else 'Outflow Dominant'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Flow Analysis Tabs
    st.markdown("### Advanced Flow Analysis")
    
    flow_tab1, flow_tab2, flow_tab3, flow_tab4 = st.tabs([
        "📊 Inflow Analysis", 
        "📉 Outflow Analysis",
        "📈 Trend Analysis",
        "📊 Bollinger Bands"
    ])
    
    with flow_tab1:
        st.markdown("##### Advanced Inflow Analysis")
        
        if inflow_categories:
            # Create multi-panel inflow analysis
            fig_inflows = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Latest Inflows by Category',
                    'Inflow Trend vs Historical Average',
                    'Inflow Contribution (%)',
                    'Inflow Volatility Analysis'
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.12,
                specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                       [{'type': 'pie'}, {'type': 'scatter'}]]
            )
            
            # 1. Latest Inflows Bar Chart
            latest_inflows = []
            for category in inflow_categories:
                latest_flow = data_dict[category]['flows'].iloc[-1, 0]
                latest_inflows.append(latest_flow)
            
            # Color by magnitude
            inflow_colors = ['#2ecc71' if x > np.median(latest_inflows) else '#27ae60' for x in latest_inflows]
            
            fig_inflows.add_trace(
                go.Bar(
                    x=inflow_categories,
                    y=latest_inflows,
                    name='Latest Inflow',
                    marker_color=inflow_colors,
                    text=[f"${x:,.0f}M" for x in latest_inflows],
                    textposition='auto',
                    hovertemplate='%{x}<br>Inflow: $%{y:,.0f}M<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Inflow Trend Analysis
            for idx, category in enumerate(inflow_categories[:3]):  # Show top 3 for clarity
                data = data_dict[category]
                flows = data['flows']['Flow']
                
                if len(flows) >= 12:
                    # Calculate moving average
                    ma_window = min(12, len(flows))
                    ma = flows.rolling(window=ma_window).mean()
                    
                    fig_inflows.add_trace(
                        go.Scatter(
                            x=flows.index,
                            y=flows,
                            name=f'{category} Flow',
                            mode='lines',
                            line=dict(width=1.5, color=data['color']),
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category}: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    # Add moving average
                    fig_inflows.add_trace(
                        go.Scatter(
                            x=ma.index,
                            y=ma,
                            name=f'{category} {ma_window}-period MA',
                            mode='lines',
                            line=dict(width=2, color=data['color'], dash='dash'),
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category} MA: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=1, col=2
                    )
            
            # 3. Inflow Contribution Pie Chart
            inflow_values = [data_dict[cat]['flows'].iloc[-1, 0] for cat in inflow_categories]
            fig_inflows.add_trace(
                go.Pie(
                    labels=inflow_categories,
                    values=inflow_values,
                    hole=0.4,
                    marker_colors=[data_dict[cat]['color'] for cat in inflow_categories],
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Inflow Volatility Analysis
            for idx, category in enumerate(inflow_categories[:2]):  # Show top 2 for clarity
                data = data_dict[category]
                flows = data['flows']['Flow']
                
                if len(flows) >= 20:
                    # Calculate rolling volatility
                    rolling_vol = flows.rolling(window=6).std()
                    
                    fig_inflows.add_trace(
                        go.Scatter(
                            x=flows.index,
                            y=rolling_vol,
                            name=f'{category} Volatility',
                            mode='lines',
                            line=dict(width=2, color=data['color']),
                            fill='tozeroy',
                            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(data["color"])) + [0.2])}',
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category} σ: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=2, col=2
                    )
            
            fig_inflows.update_layout(
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_inflows, use_container_width=True)
            
            # Inflow Statistics
            st.markdown("##### Inflow Statistics")
            
            inflow_stats = []
            for category in inflow_categories:
                data = data_dict[category]
                flows = data['flows']['Flow']
                
                if len(flows) > 0:
                    latest = flows.iloc[-1]
                    avg = flows.mean()
                    std = flows.std()
                    max_flow = flows.max()
                    min_flow = flows.min()
                    
                    inflow_stats.append({
                        'Category': category,
                        'Latest': f"${latest:,.0f}M",
                        'Average': f"${avg:,.0f}M",
                        'Std Dev': f"${std:,.0f}M",
                        'Max': f"${max_flow:,.0f}M",
                        'Min': f"${min_flow:,.0f}M",
                        'Volatility': f"{std/avg*100:.1f}%" if avg > 0 else "N/A"
                    })
            
            if inflow_stats:
                inflow_df = pd.DataFrame(inflow_stats)
                st.dataframe(inflow_df, use_container_width=True, height=200)
        else:
            st.info("No inflows in latest period")
    
    with flow_tab2:
        st.markdown("##### Advanced Outflow Analysis")
        
        if outflow_categories:
            # Create multi-panel outflow analysis
            fig_outflows = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Latest Outflows by Category',
                    'Outflow Trend vs Historical Average',
                    'Outflow Contribution (%)',
                    'Outflow Volatility Analysis'
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.12,
                specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                       [{'type': 'pie'}, {'type': 'scatter'}]]
            )
            
            # 1. Latest Outflows Bar Chart
            latest_outflows = []
            for category in outflow_categories:
                latest_flow = abs(data_dict[category]['flows'].iloc[-1, 0])
                latest_outflows.append(latest_flow)
            
            # Color by magnitude
            outflow_colors = ['#e74c3c' if x > np.median(latest_outflows) else '#c0392b' for x in latest_outflows]
            
            fig_outflows.add_trace(
                go.Bar(
                    x=outflow_categories,
                    y=latest_outflows,
                    name='Latest Outflow',
                    marker_color=outflow_colors,
                    text=[f"${x:,.0f}M" for x in latest_outflows],
                    textposition='auto',
                    hovertemplate='%{x}<br>Outflow: $%{y:,.0f}M<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Outflow Trend Analysis
            for idx, category in enumerate(outflow_categories[:3]):
                data = data_dict[category]
                flows = abs(data['flows']['Flow'])  # Use absolute values for outflow analysis
                
                if len(flows) >= 12:
                    # Calculate moving average
                    ma_window = min(12, len(flows))
                    ma = flows.rolling(window=ma_window).mean()
                    
                    fig_outflows.add_trace(
                        go.Scatter(
                            x=flows.index,
                            y=flows,
                            name=f'{category} Outflow',
                            mode='lines',
                            line=dict(width=1.5, color=data['color']),
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category}: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    # Add moving average
                    fig_outflows.add_trace(
                        go.Scatter(
                            x=ma.index,
                            y=ma,
                            name=f'{category} {ma_window}-period MA',
                            mode='lines',
                            line=dict(width=2, color=data['color'], dash='dash'),
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category} MA: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=1, col=2
                    )
            
            # 3. Outflow Contribution Pie Chart
            outflow_values = [abs(data_dict[cat]['flows'].iloc[-1, 0]) for cat in outflow_categories]
            fig_outflows.add_trace(
                go.Pie(
                    labels=outflow_categories,
                    values=outflow_values,
                    hole=0.4,
                    marker_colors=[data_dict[cat]['color'] for cat in outflow_categories],
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Outflow Volatility Analysis
            for idx, category in enumerate(outflow_categories[:2]):
                data = data_dict[category]
                flows = abs(data['flows']['Flow'])
                
                if len(flows) >= 20:
                    # Calculate rolling volatility
                    rolling_vol = flows.rolling(window=6).std()
                    
                    fig_outflows.add_trace(
                        go.Scatter(
                            x=flows.index,
                            y=rolling_vol,
                            name=f'{category} Volatility',
                            mode='lines',
                            line=dict(width=2, color=data['color']),
                            fill='tozeroy',
                            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(data["color"])) + [0.2])}',
                            showlegend=True,
                            hovertemplate='%{x|%b %Y}<br>' + f'{category} σ: $%{{y:,.0f}}M<extra></extra>'
                        ),
                        row=2, col=2
                    )
            
            fig_outflows.update_layout(
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_outflows, use_container_width=True)
            
            # Outflow Statistics
            st.markdown("##### Outflow Statistics")
            
            outflow_stats = []
            for category in outflow_categories:
                data = data_dict[category]
                flows = abs(data['flows']['Flow'])
                
                if len(flows) > 0:
                    latest = flows.iloc[-1]
                    avg = flows.mean()
                    std = flows.std()
                    max_flow = flows.max()
                    min_flow = flows.min()
                    
                    outflow_stats.append({
                        'Category': category,
                        'Latest': f"${latest:,.0f}M",
                        'Average': f"${avg:,.0f}M",
                        'Std Dev': f"${std:,.0f}M",
                        'Max': f"${max_flow:,.0f}M",
                        'Min': f"${min_flow:,.0f}M",
                        'Volatility': f"{std/avg*100:.1f}%" if avg > 0 else "N/A"
                    })
            
            if outflow_stats:
                outflow_df = pd.DataFrame(outflow_stats)
                st.dataframe(outflow_df, use_container_width=True, height=200)
        else:
            st.info("No outflows in latest period")
    
    with flow_tab3:
        st.markdown("##### Trend Analysis - Above/Below Trend Indicators")
        
        # Configuration for trend analysis
        col1, col2 = st.columns(2)
        with col1:
            trend_window = st.slider("Trend Window (periods)", 4, 24, 12, key="trend_window")
        
        with col2:
            threshold_multiplier = st.slider("Threshold Multiplier", 0.5, 3.0, 1.0, 0.1, key="trend_threshold")
        
        # Perform trend analysis for each category
        trend_results = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow']
                
                if len(flows) >= trend_window:
                    # Calculate trend components
                    moving_avg = flows.rolling(window=trend_window).mean()
                    moving_std = flows.rolling(window=trend_window).std()
                    
                    latest_flow = flows.iloc[-1]
                    latest_ma = moving_avg.iloc[-1]
                    latest_std = moving_std.iloc[-1]
                    
                    # Calculate z-score
                    z_score = (latest_flow - latest_ma) / latest_std if latest_std > 0 else 0
                    
                    # Determine trend status
                    if latest_flow > latest_ma + (latest_std * threshold_multiplier):
                        trend_status = "Well Above Trend"
                        status_color = "#27ae60"
                        status_icon = "📈"
                    elif latest_flow > latest_ma:
                        trend_status = "Above Trend"
                        status_color = "#2ecc71"
                        status_icon = "↗️"
                    elif latest_flow < latest_ma - (latest_std * threshold_multiplier):
                        trend_status = "Well Below Trend"
                        status_color = "#e74c3c"
                        status_icon = "📉"
                    elif latest_flow < latest_ma:
                        trend_status = "Below Trend"
                        status_color = "#c0392b"
                        status_icon = "↘️"
                    else:
                        trend_status = "At Trend"
                        status_color = "#3498db"
                        status_icon = "➡️"
                    
                    # Calculate trend strength
                    trend_strength = abs(z_score)
                    
                    if trend_strength > 2:
                        strength_label = "Very Strong"
                        strength_color = "#e74c3c"
                    elif trend_strength > 1.5:
                        strength_label = "Strong"
                        strength_color = "#f39c12"
                    elif trend_strength > 1:
                        strength_label = "Moderate"
                        strength_color = "#f1c40f"
                    elif trend_strength > 0.5:
                                           else:
                        strength_label = "Weak"
                        strength_color = "#27ae60"
                    
                    trend_results.append({
                        'Category': category,
                        'Latest Flow': f"${latest_flow:,.0f}M",
                        f'{trend_window}-Period MA': f"${latest_ma:,.0f}M",
                        'Z-Score': f"{z_score:.2f}",
                        'Trend Status': f"{status_icon} {trend_status}",
                        'Trend Strength': f"<span style='color:{strength_color}'>{strength_label}</span>",
                        'Status Color': status_color
                    })
        
        # Display trend results
        if trend_results:
            # Create trend indicators
            cols = st.columns(len(trend_results))
            for idx, result in enumerate(trend_results):
                with cols[idx]:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {result["Status Color"]}'>
                        <div class='metric-label'>{result['Category']}</div>
                        <div class='metric-value'>{result['Trend Status'].split()[-1]}</div>
                        <div style='font-size: 0.9rem; margin: 0.5rem 0;'>
                            <strong>Z-Score:</strong> {result['Z-Score']}<br>
                            <strong>Flow:</strong> {result['Latest Flow']}<br>
                            <strong>Trend MA:</strong> {result[f'{trend_window}-Period MA']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display trend table
            st.markdown("##### Detailed Trend Analysis")
            trend_df = pd.DataFrame(trend_results)
            # Remove the color column from display
            display_df = trend_df.drop(columns=['Status Color'])
            st.dataframe(display_df, use_container_width=True, height=200)
            
            # Visual trend chart
            st.markdown("##### Visual Trend Analysis")
            fig_trend = go.Figure()
            
            for category, data in data_dict.items():
                if 'flows' in data and not data['flows'].empty:
                    flows = data['flows']['Flow']
                    color = data.get('color', PROFESSIONAL_COLORS[0])
                    
                    # Add actual flow
                    fig_trend.add_trace(go.Scatter(
                        x=flows.index,
                        y=flows,
                        name=f'{category} Flow',
                        mode='lines',
                        line=dict(width=1.5, color=color),
                        hovertemplate='%{x|%b %Y}<br>' + f'{category}: $%{{y:,.0f}}M<extra></extra>'
                    ))
                    
                    # Add moving average
                    if len(flows) >= trend_window:
                        moving_avg = flows.rolling(window=trend_window).mean()
                        fig_trend.add_trace(go.Scatter(
                            x=moving_avg.index,
                            y=moving_avg,
                            name=f'{category} {trend_window}-Period MA',
                            mode='lines',
                            line=dict(width=2, color=color, dash='dash'),
                            hovertemplate='%{x|%b %Y}<br>' + f'{category} MA: $%{{y:,.0f}}M<extra></extra>'
                        ))
            
            fig_trend.update_layout(
                title=f'Flow Trends with {trend_window}-Period Moving Average',
                xaxis_title='Date',
                yaxis_title='Flow ($M)',
                height=500,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data for trend analysis")
    
    with flow_tab4:
        st.markdown("##### Bollinger Band Analysis")
        
        # Bollinger Band configuration
        col1, col2 = st.columns(2)
        with col1:
            bb_window = st.slider("Bollinger Window (periods)", 5, 30, 20, key="bb_window")
        
        with col2:
            bb_std = st.slider("Standard Deviation Multiplier", 1.0, 3.0, 2.0, 0.1, key="bb_std")
        
        # Let user select a category for detailed Bollinger analysis
        if data_dict:
            selected_category = st.selectbox(
                "Select Category for Bollinger Band Analysis",
                list(data_dict.keys()),
                key="bb_category"
            )
            
            if selected_category:
                data = data_dict[selected_category]
                if 'flows' in data and not data['flows'].empty:
                    flows = data['flows']['Flow']
                    color = data.get('color', PROFESSIONAL_COLORS[0])
                    
                    if len(flows) >= bb_window:
                        # Calculate Bollinger Bands
                        rolling_mean = flows.rolling(window=bb_window).mean()
                        rolling_std = flows.rolling(window=bb_window).std()
                        
                        upper_band = rolling_mean + (rolling_std * bb_std)
                        lower_band = rolling_mean - (rolling_std * bb_std)
                        
                        # Create Bollinger Band chart
                        fig_bb = go.Figure()
                        
                        # Add actual flow
                        fig_bb.add_trace(go.Scatter(
                            x=flows.index,
                            y=flows,
                            name='Actual Flow',
                            mode='lines',
                            line=dict(width=2, color=color),
                            hovertemplate='%{x|%b %Y}<br>Actual: $%{y:,.0f}M<extra></extra>'
                        ))
                        
                        # Add moving average
                        fig_bb.add_trace(go.Scatter(
                            x=rolling_mean.index,
                            y=rolling_mean,
                            name=f'{bb_window}-Period MA',
                            mode='lines',
                            line=dict(width=1.5, color='#2c3e50', dash='dash'),
                            hovertemplate='%{x|%b %Y}<br>MA: $%{y:,.0f}M<extra></extra>'
                        ))
                        
                        # Add upper band
                        fig_bb.add_trace(go.Scatter(
                            x=upper_band.index,
                            y=upper_band,
                            name=f'Upper Band ({bb_std}σ)',
                            mode='lines',
                            line=dict(width=1, color='#e74c3c', dash='dot'),
                            hovertemplate='%{x|%b %Y}<br>Upper: $%{y:,.0f}M<extra></extra>'
                        ))
                        
                        # Add lower band
                        fig_bb.add_trace(go.Scatter(
                            x=lower_band.index,
                            y=lower_band,
                            name=f'Lower Band ({bb_std}σ)',
                            mode='lines',
                            line=dict(width=1, color='#27ae60', dash='dot'),
                            hovertemplate='%{x|%b %Y}<br>Lower: $%{y:,.0f}M<extra></extra>'
                        ))
                        
                        # Fill between bands
                        fig_bb.add_trace(go.Scatter(
                            x=pd.concat([upper_band.index, lower_band.index[::-1]]),
                            y=pd.concat([upper_band, lower_band[::-1]]),
                            fill='toself',
                            fillcolor='rgba(52, 152, 219, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Bollinger Band',
                            showlegend=False,
                            hovertemplate='%{x|%b %Y}<br>Band Width<extra></extra>'
                        ))
                        
                        # Calculate band width and statistics
                        latest_flow = flows.iloc[-1]
                        latest_upper = upper_band.iloc[-1]
                        latest_lower = lower_band.iloc[-1]
                        latest_ma = rolling_mean.iloc[-1]
                        
                        # Determine position relative to bands
                        if latest_flow > latest_upper:
                            position = "Above Upper Band"
                            position_color = "#e74c3c"
                            position_icon = "🚨"
                            signal = "Overbought/Extreme Inflow"
                        elif latest_flow < latest_lower:
                            position = "Below Lower Band"
                            position_color = "#27ae60"
                            position_icon = "📉"
                            signal = "Oversold/Extreme Outflow"
                        elif latest_flow > latest_ma:
                            position = "Above MA, Within Bands"
                            position_color = "#2ecc71"
                            position_icon = "↗️"
                            signal = "Moderate Inflow"
                        else:
                            position = "Below MA, Within Bands"
                            position_color = "#f39c12"
                            position_icon = "↘️"
                            signal = "Moderate Outflow"
                        
                        # Calculate %B indicator
                        if latest_upper != latest_lower:
                            percent_b = (latest_flow - latest_lower) / (latest_upper - latest_lower) * 100
                        else:
                            percent_b = 50
                        
                        # Band width
                        band_width = ((latest_upper - latest_lower) / latest_ma * 100) if latest_ma != 0 else 0
                        
                        fig_bb.update_layout(
                            title=f'{selected_category} Bollinger Band Analysis',
                            xaxis_title='Date',
                            yaxis_title='Flow ($M)',
                            height=500,
                            hovermode='x unified',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_bb, use_container_width=True)
                        
                        # Display Bollinger Band metrics
                        st.markdown("##### Bollinger Band Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Position Relative to Bands",
                                f"{position_icon} {position}",
                                help="Current flow position relative to Bollinger Bands"
                            )
                        
                        with col2:
                            st.metric(
                                "%B Indicator",
                                f"{percent_b:.1f}%",
                                delta=f"{'High' if percent_b > 80 else 'Low' if percent_b < 20 else 'Neutral'}",
                                delta_color="normal",
                                help="0% = at lower band, 100% = at upper band, 50% = at middle band"
                            )
                        
                        with col3:
                            st.metric(
                                "Band Width",
                                f"{band_width:.1f}%",
                                help="Volatility indicator: Higher width = higher volatility"
                            )
                        
                        with col4:
                            st.metric(
                                "Signal",
                                signal,
                                help="Trading/investment signal based on Bollinger Band position"
                            )
                        
                        # Display band statistics
                        st.markdown("##### Band Statistics")
                        
                        bb_stats = pd.DataFrame({
                            'Metric': ['Upper Band', 'Middle Band (MA)', 'Lower Band', 
                                      'Current Flow', 'Band Distance (Upper)', 'Band Distance (Lower)'],
                            'Value': [f"${latest_upper:,.0f}M", f"${latest_ma:,.0f}M", f"${latest_lower:,.0f}M",
                                     f"${latest_flow:,.0f}M", 
                                     f"{(latest_flow - latest_upper)/latest_ma*100:.1f}%" if latest_ma != 0 else "N/A",
                                     f"{(latest_lower - latest_flow)/latest_ma*100:.1f}%" if latest_ma != 0 else "N/A"],
                            'Description': ['Upper boundary', 'Moving average center line', 'Lower boundary',
                                          'Latest actual flow', '% above upper band', '% below lower band']
                        })
                        
                        st.dataframe(bb_stats, use_container_width=True, height=200)
                        
                        # Historical band analysis
                        st.markdown("##### Historical Band Analysis")
                        
                        # Calculate how often flow is outside bands
                        outside_upper = (flows > upper_band).sum()
                        outside_lower = (flows < lower_band).sum()
                        total_periods = len(flows.dropna())
                        
                        if total_periods > 0:
                            percent_upper = outside_upper / total_periods * 100
                            percent_lower = outside_lower / total_periods * 100
                            percent_inside = 100 - percent_upper - percent_lower
                            
                            # Create historical analysis
                            hist_fig = go.Figure(data=[
                                go.Bar(
                                    name='% Time Above Upper Band',
                                    x=['Historical Distribution'],
                                    y=[percent_upper],
                                    marker_color='#e74c3c',
                                    text=f'{percent_upper:.1f}%',
                                    textposition='auto'
                                ),
                                go.Bar(
                                    name='% Time Within Bands',
                                    x=['Historical Distribution'],
                                    y=[percent_inside],
                                    marker_color='#3498db',
                                    text=f'{percent_inside:.1f}%',
                                    textposition='auto'
                                ),
                                go.Bar(
                                    name='% Time Below Lower Band',
                                    x=['Historical Distribution'],
                                    y=[percent_lower],
                                    marker_color='#27ae60',
                                    text=f'{percent_lower:.1f}%',
                                    textposition='auto'
                                )
                            ])
                            
                            hist_fig.update_layout(
                                title='Historical Distribution of Flow Relative to Bands',
                                barmode='stack',
                                height=400,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                yaxis=dict(
                                    title='Percentage of Time',
                                    ticksuffix='%',
                                    range=[0, 100]
                                )
                            )
                            
                            st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        st.warning(f"Need at least {bb_window} periods of data for Bollinger Band analysis")
                else:
                    st.warning("No flow data available for selected category")
        else:
            st.info("No data available for Bollinger Band analysis")

def create_correlation_matrix(data_dict):
    """Create correlation analysis between different fund categories"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Correlation Analysis</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    # Extract flow data
    flow_data = {}
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flow_data[category] = data['flows']['Flow']
    
    if len(flow_data) < 2:
        st.info("Need at least 2 categories for correlation analysis")
        return
    
    # Create correlation matrix
    df = pd.DataFrame(flow_data)
    correlation_matrix = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Flow Correlation Matrix',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("##### Correlation Insights")
    
    # Find strongest correlations
    insights = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.3:  # Only show meaningful correlations
                cat1 = correlation_matrix.columns[i]
                cat2 = correlation_matrix.columns[j]
                
                if corr > 0.7:
                    strength = "Very Strong Positive"
                    color = "#27ae60"
                    icon = "📈📈"
                    implication = "Likely similar investor behavior or market factors"
                elif corr > 0.5:
                    strength = "Strong Positive"
                    color = "#2ecc71"
                    icon = "📈"
                    implication = "Some common driving factors"
                elif corr > 0.3:
                    strength = "Moderate Positive"
                    color = "#f1c40f"
                    icon = "↗️"
                    implication = "Mild relationship"
                elif corr < -0.7:
                    strength = "Very Strong Negative"
                    color = "#e74c3c"
                    icon = "📉📉"
                    implication = "Opposing investor behavior (flight between categories)"
                elif corr < -0.5:
                    strength = "Strong Negative"
                    color = "#c0392b"
                    icon = "📉"
                    implication = "Inverse relationship likely"
                elif corr < -0.3:
                    strength = "Moderate Negative"
                    color = "#e67e22"
                    icon = "↘️"
                    implication = "Some inverse relationship"
                else:
                    continue
                
                insights.append({
                    'Relationship': f"{cat1} ↔ {cat2}",
                    'Correlation': f"{corr:.2f}",
                    'Strength': f"<span style='color:{color}'>{icon} {strength}</span>",
                    'Implication': implication
                })
    
    if insights:
        insights_df = pd.DataFrame(insights)
        st.dataframe(insights_df, use_container_width=True, height=200)
    else:
        st.info("No strong correlations found (|r| > 0.3)")
    
    # Rolling correlation analysis
    st.markdown("##### Rolling Correlation Analysis")
    
    rolling_window = st.slider("Rolling Window Size", 10, 50, 20, key="corr_window")
    
    # Create rolling correlation visualization
    if len(df) > rolling_window:
        fig_rolling = go.Figure()
        
        # Select two main categories for rolling correlation
        if len(df.columns) >= 2:
            cat1, cat2 = df.columns[0], df.columns[1]
            rolling_corr = df[cat1].rolling(window=rolling_window).corr(df[cat2])
            
            fig_rolling.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode='lines',
                name=f'{cat1} vs {cat2}',
                line=dict(width=2, color='#3498db'),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)',
                hovertemplate='%{x|%b %Y}<br>Correlation: %{y:.2f}<extra></extra>'
            ))
            
            fig_rolling.update_layout(
                title=f'{rolling_window}-Period Rolling Correlation: {cat1} vs {cat2}',
                xaxis_title='Date',
                yaxis_title='Correlation',
                height=400,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(range=[-1, 1])
            )
            
            # Add horizontal lines for reference
            fig_rolling.add_hline(y=0.7, line_dash="dot", line_color="#27ae60", 
                                 annotation_text="Strong Positive", annotation_position="top right")
            fig_rolling.add_hline(y=0, line_dash="solid", line_color="#666666", opacity=0.5)
            fig_rolling.add_hline(y=-0.7, line_dash="dot", line_color="#e74c3c", 
                                 annotation_text="Strong Negative", annotation_position="bottom right")
            
            st.plotly_chart(fig_rolling, use_container_width=True)
    else:
        st.warning(f"Need at least {rolling_window} periods of data for rolling correlation")

def create_advanced_analytics(data_dict):
    """Create advanced analytics section"""
    latest_date = get_latest_date_info(data_dict)
    
    st.markdown(f"""
    <div class="section-header">
        <span>Advanced Analytics</span>
        <span class="date-indicator">Latest Data: {latest_date}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not data_dict:
        st.warning("No data available")
        return
    
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "📊 Volatility Analysis",
        "📈 Seasonality Analysis",
        "🔍 Statistical Insights"
    ])
    
    with analytics_tab1:
        st.markdown("##### Volatility Analysis")
        
        # Calculate and display volatility metrics
        volatility_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows']['Flow']
                returns = flows.pct_change().dropna()
                
                if len(returns) > 0:
                    # Calculate various volatility metrics
                    daily_vol = returns.std() * 100
                    annualized_vol = daily_vol * np.sqrt(252)  # Annualization
                    max_drawdown = (flows / flows.cummax() - 1).min() * 100
                    var_95 = np.percentile(returns, 5) * 100
                    var_99 = np.percentile(returns, 1) * 100
                    
                    volatility_data.append({
                        'Category': category,
                        'Daily Vol (%)': f"{daily_vol:.2f}",
                        'Annual Vol (%)': f"{annualized_vol:.2f}",
                        'Max Drawdown (%)': f"{max_drawdown:.2f}",
                        'VaR 95% (%)': f"{var_95:.2f}",
                        'VaR 99% (%)': f"{var_99:.2f}",
                        'Risk Level': 'High' if annualized_vol > 30 else 'Medium' if annualized_vol > 15 else 'Low'
                    })
        
        if volatility_data:
            volatility_df = pd.DataFrame(volatility_data)
            st.dataframe(volatility_df, use_container_width=True)
            
            # Volatility comparison chart
            st.markdown("##### Volatility Comparison")
            
            fig_vol = go.Figure()
            categories = [d['Category'] for d in volatility_data]
            annual_vols = [float(d['Annual Vol (%)']) for d in volatility_data]
            
            # Color by risk level
            colors = []
            for vol in annual_vols:
                if vol > 30:
                    colors.append('#e74c3c')
                elif vol > 15:
                    colors.append('#f39c12')
                else:
                    colors.append('#27ae60')
            
            fig_vol.add_trace(go.Bar(
                x=categories,
                y=annual_vols,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in annual_vols],
                textposition='auto',
                hovertemplate='%{x}<br>Annual Volatility: %{y:.1f}%<extra></extra>'
            ))
            
            fig_vol.update_layout(
                title='Annualized Volatility by Category',
                xaxis_title='Category',
                yaxis_title='Annual Volatility (%)',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add risk level bands
            fig_vol.add_hrect(y0=0, y1=15, line_width=0, fillcolor="green", opacity=0.1)
            fig_vol.add_hrect(y0=15, y1=30, line_width=0, fillcolor="yellow", opacity=0.1)
            fig_vol.add_hrect(y0=30, y1=max(annual_vols)*1.1, line_width=0, fillcolor="red", opacity=0.1)
            
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with analytics_tab2:
        st.markdown("##### Seasonality Analysis")
        
        # Let user select a category
        if data_dict:
            selected_category = st.selectbox(
                "Select Category for Seasonality Analysis",
                list(data_dict.keys()),
                key="seasonality_category"
            )
            
            if selected_category:
                data = data_dict[selected_category]
                if 'flows' in data and not data['flows'].empty:
                    flows = data['flows']['Flow']
                    
                    if len(flows) >= 24:  # Need at least 2 years for seasonality
                        # Prepare data for seasonality analysis
                        flows_df = flows.reset_index()
                        flows_df.columns = ['Date', 'Flow']
                        flows_df['Year'] = flows_df['Date'].dt.year
                        flows_df['Month'] = flows_df['Date'].dt.month
                        flows_df['Month_Name'] = flows_df['Date'].dt.strftime('%b')
                        
                        # Monthly average
                        monthly_avg = flows_df.groupby(['Month', 'Month_Name'])['Flow'].mean().reset_index()
                        monthly_avg = monthly_avg.sort_values('Month')
                        
                        # Create seasonality chart
                        fig_season = go.Figure()
                        
                        fig_season.add_trace(go.Bar(
                            x=monthly_avg['Month_Name'],
                            y=monthly_avg['Flow'],
                            marker_color='#3498db',
                            text=[f"${x:,.0f}M" for x in monthly_avg['Flow']],
                            textposition='auto',
                            name='Average Monthly Flow',
                            hovertemplate='%{x}<br>Average Flow: $%{y:,.0f}M<extra></extra>'
                        ))
                        
                        fig_season.update_layout(
                            title=f'{selected_category} - Monthly Seasonality Pattern',
                            xaxis_title='Month',
                            yaxis_title='Average Flow ($M)',
                            height=400,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_season, use_container_width=True)
                        
                        # Year-over-year comparison
                        st.markdown("##### Year-over-Year Comparison")
                        
                        if len(flows_df['Year'].unique()) >= 2:
                            # Pivot for year-over-year comparison
                            yearly_pivot = flows_df.pivot_table(
                                index='Month_Name',
                                columns='Year',
                                values='Flow',
                                aggfunc='mean'
                            )
                            
                            # Order by month
                            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            yearly_pivot = yearly_pivot.reindex(month_order)
                            
                            fig_yoy = go.Figure()
                            
                            for year in yearly_pivot.columns:
                                fig_yoy.add_trace(go.Scatter(
                                    x=yearly_pivot.index,
                                    y=yearly_pivot[year],
                                    mode='lines+markers',
                                    name=str(year),
                                    hovertemplate='%{x} %{text}<br>Flow: $%{y:,.0f}M<extra></extra>',
                                    text=[str(year)] * len(yearly_pivot)
                                ))
                            
                            fig_yoy.update_layout(
                                title=f'{selected_category} - Year-over-Year Monthly Comparison',
                                xaxis_title='Month',
                                yaxis_title='Average Flow ($M)',
                                height=400,
                                hovermode='x unified',
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig_yoy, use_container_width=True)
                    else:
                        st.warning("Need at least 24 periods (2 years) for seasonality analysis")
                else:
                    st.warning("No flow data available for selected category")
    
    with analytics_tab3:
        st.markdown("##### Statistical Insights")
        
        if data_dict:
            # Let user select a category for detailed statistics
            selected_category = st.selectbox(
                "Select Category for Statistical Analysis",
                list(data_dict.keys()),
                key="stats_category"
            )
            
            if selected_category:
                data = data_dict[selected_category]
                if 'flows' in data and not data['flows'].empty:
                    flows = data['flows']['Flow'].dropna()
                    
                    if len(flows) > 0:
                        # Calculate statistics
                        stats_summary = {
                            'Number of Observations': len(flows),
                            'Mean': f"${flows.mean():,.0f}M",
                            'Median': f"${flows.median():,.0f}M",
                            'Standard Deviation': f"${flows.std():,.0f}M",
                            'Minimum': f"${flows.min():,.0f}M",
                            'Maximum': f"${flows.max():,.0f}M",
                            'Skewness': f"{flows.skew():.2f}",
                            'Kurtosis': f"{flows.kurtosis():.2f}",
                            '25th Percentile': f"${flows.quantile(0.25):,.0f}M",
                            '75th Percentile': f"${flows.quantile(0.75):,.0f}M"
                        }
                        
                        # Display statistics
                        stats_df = pd.DataFrame(list(stats_summary.items()), columns=['Statistic', 'Value'])
                        st.dataframe(stats_df, use_container_width=True, height=400)
                        
                        # Distribution analysis
                        st.markdown("##### Distribution Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig_hist = px.histogram(
                                x=flows,
                                nbins=30,
                                title=f'{selected_category} Flow Distribution',
                                labels={'x': 'Flow ($M)', 'y': 'Frequency'},
                                color_discrete_sequence=['#3498db']
                            )
                            
                            fig_hist.update_layout(
                                height=400,
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Q-Q plot
                            from scipy import stats
                            import numpy as np
                            
                            qq = stats.probplot(flows, dist="norm")
                            x = np.array([qq[0][0][0], qq[0][0][-1]])
                            
                            fig_qq = go.Figure()
                            
                            fig_qq.add_trace(go.Scatter(
                                x=qq[0][0],
                                y=qq[0][1],
                                mode='markers',
                                name='Actual',
                                marker=dict(color='#3498db', size=8)
                            ))
                            
                            fig_qq.add_trace(go.Scatter(
                                x=x,
                                y=qq[1][1] + qq[1][0] * x,
                                mode='lines',
                                name='Normal',
                                line=dict(color='#e74c3c', dash='dash')
                            ))
                            
                            fig_qq.update_layout(
                                title='Q-Q Plot (Normal Distribution Test)',
                                xaxis_title='Theoretical Quantiles',
                                yaxis_title='Sample Quantiles',
                                height=400,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_qq, use_container_width=True)
                        
                        # Normality test
                        from scipy.stats import shapiro
                        
                        if len(flows) <= 5000:  # Shapiro test limitation
                            stat, p_value = shapiro(flows)
                            
                            st.markdown(f"""
                            ##### Normality Test Results
                            - **Shapiro-Wilk Test Statistic**: {stat:.4f}
                            - **p-value**: {p_value:.4f}
                            - **Interpretation**: {'Data appears normal (fail to reject H0)' if p_value > 0.05 else 'Data does not appear normal (reject H0)'}
                            """)

def main():
    """Main application function"""
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Data frequency
        frequency = st.selectbox(
            "Data Frequency",
            ["monthly", "weekly"],
            help="Select data frequency"
        )
        
        # Date range
        years_back = st.slider(
            "Analysis Period (Years)",
            1, 10, 5,
            help="Number of years of historical data to analyze"
        )
        
        start_date = (datetime.today() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
        
        # Fund category selection
        st.markdown("### 📊 Fund Categories")
        selected_categories = []
        for category in FRED_SERIES.keys():
            if st.checkbox(category, value=True if category in ['Total Mutual Fund Assets', 'Equity Funds', 'Bond Funds'] else False):
                selected_categories.append(category)
        
        if not selected_categories:
            st.warning("Please select at least one fund category")
            return
        
        # Analysis settings
        st.markdown("### 🔧 Analysis Settings")
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        show_correlation = st.checkbox("Show Correlation Analysis", value=True)
        
        # Info
        st.markdown("---")
        st.markdown("""
        ### 📈 About This Dashboard
        
        This Institutional Fund Flow Analytics dashboard provides:
        
        - **Real-time fund flow analysis**
        - **Advanced trend detection**
        - **Correlation insights**
        - **Risk and volatility metrics**
        - **Professional visualization**
        
        Data is simulated for demonstration.
        """)
    
    # Load data
    with st.spinner("Loading fund flow data..."):
        data_dict = load_fund_data(selected_categories, start_date, frequency)
    
    if not data_dict:
        st.error("Failed to load data. Please check your configuration.")
        return
    
    # Main dashboard tabs
    main_tabs = st.tabs([
        "📈 Executive Overview",
        "📊 Growth Analysis",
        "💰 Flow Dynamics",
        "📈 Advanced Analytics",
        "🔗 Correlation Matrix"
    ])
    
    with main_tabs[0]:
        create_executive_summary(data_dict, frequency)
        
        # Quick insights
        st.markdown("### 🎯 Key Insights")
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.markdown("""
            <div class='analysis-card'>
                <strong>💰 Flow Direction</strong><br>
                Track inflow/outflow patterns across categories to identify market sentiment shifts.
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class='analysis-card'>
                <strong>📈 Trend Strength</strong><br>
                Measure flow persistence using z-scores and moving average analysis.
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col3:
            st.markdown("""
            <div class='analysis-card'>
                <strong>🎯 Market Timing</strong><br>
                Use Bollinger Bands to identify overbought/oversold conditions.
            </div>
            """, unsafe_allow_html=True)
    
    with main_tabs[1]:
        create_professional_growth_charts(data_dict)
    
    with main_tabs[2]:
        create_enhanced_flow_analysis(data_dict, frequency)
    
    with main_tabs[3]:
        if show_advanced:
            create_advanced_analytics(data_dict)
        else:
            st.info("Enable 'Show Advanced Analytics' in the sidebar to access this section")
    
    with main_tabs[4]:
        if show_correlation:
            create_correlation_matrix(data_dict)
        else:
            st.info("Enable 'Show Correlation Analysis' in the sidebar to access this section")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Institutional Fund Flow Analytics Dashboard v2.0 | Professional Edition</p>
        <p>Data Source: Simulated FRED-style Data | Analysis Period: Customizable</p>
        <p style='font-size: 0.75rem; color: #999999; margin-top: 1rem;'>
            This dashboard is for institutional analysis and research purposes only.
            All data shown is simulated for demonstration.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
