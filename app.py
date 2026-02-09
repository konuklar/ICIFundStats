import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_datareader.data as web
import warnings
from scipy import stats
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Institutional Fund Flow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, clean CSS with neutral color scheme
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
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .metric-change {
        font-size: 0.85rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 5px;
    }
    .positive-change {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .negative-change {
        background-color: #ffebee;
        color: #c62828;
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
    .analysis-section {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2c3e50;
    }
    .data-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
        background: #e9ecef;
        color: #495057;
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
st.markdown('<p class="sub-header">Professional Analysis of Mutual Fund & ETF Flows using Federal Reserve Economic Data (FRED)</p>', unsafe_allow_html=True)

# FRED Series IDs - Updated with working series
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'monthly': 'TOTMFS',
        'weekly': 'H8B3092NCBA',
        'description': 'Total Mutual Fund Assets',
        'color': '#2c3e50'
    },
    'Money Market Funds': {
        'monthly': 'MMMFFAQ027S',
        'weekly': 'H6BMMTNA',
        'description': 'Money Market Fund Assets',
        'color': '#3498db'
    },
    'Equity Funds': {
        'monthly': 'TOTCI',
        'weekly': 'H8B3053NCBA',
        'description': 'Equity Mutual Fund Assets',
        'color': '#27ae60'
    },
    'Bond Funds': {
        'monthly': 'TBCI',
        'weekly': 'H8B3094NCBA',
        'description': 'Bond/Income Fund Assets',
        'color': '#e74c3c'
    },
    'Municipal Bond Funds': {
        'monthly': 'MBCI',
        'weekly': 'H8B3095NCBA',
        'description': 'Municipal Bond Fund Assets',
        'color': '#9b59b6'
    },
    'Government Bond Funds': {
        'monthly': 'H8B3093NCBA',
        'weekly': 'H8B3093NCBA',
        'description': 'Government Bond Fund Assets',
        'color': '#f39c12'
    }
}

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date, end_date):
    """Fetch data from FRED with robust error handling"""
    try:
        df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
        return df
    except Exception as e:
        st.warning(f"Could not fetch {series_id}: {str(e)[:100]}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_data(selected_categories, start_date, frequency):
    """Load all selected FRED data"""
    data_dict = {}
    end_date = datetime.today()
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, (category, info) in enumerate(selected_categories.items()):
        progress_text.text(f"Loading {category} data...")
        
        series_id = info.get(frequency)
        if series_id:
            df = fetch_fred_data(series_id, start_date, end_date)
            
            if not df.empty:
                # Calculate flows
                df_flows = df.diff()
                df_flows.columns = [f'Flow']
                
                # Store data
                data_dict[category] = {
                    'assets': df,
                    'flows': df_flows,
                    'description': info['description'],
                    'color': info['color']
                }
        
        progress_bar.progress((idx + 1) / len(selected_categories))
    
    progress_text.empty()
    progress_bar.empty()
    
    return data_dict

def create_executive_summary(data_dict, frequency):
    """Create clean executive summary"""
    st.markdown("## Executive Summary")
    
    cols = st.columns(4)
    summary_data = []
    
    for idx, (category, data) in enumerate(list(data_dict.items())[:4]):
        with cols[idx]:
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0]
                avg_flow = data['flows'].mean().iloc[0]
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{category}</div>
                    <div class='metric-value'>${abs(latest_flow):,.0f}M</div>
                    <div class='metric-change {'positive-change' if latest_flow > 0 else 'negative-change'}'>
                        {'+' if latest_flow > 0 else ''}{latest_flow:,.0f}M
                    </div>
                    <div style='font-size: 0.8rem; color: #666666; margin-top: 8px;'>
                        Avg: ${avg_flow:,.0f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                summary_data.append({
                    'Category': category,
                    f'Latest {frequency}': f"${latest_flow:,.0f}M",
                    'Direction': 'Inflow' if latest_flow > 0 else 'Outflow',
                    '12M Avg': f"${avg_flow:,.0f}M"
                })
    
    if summary_data:
        st.markdown("### Performance Overview")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

def create_professional_growth_charts(data_dict):
    """Create clean, professional growth analysis"""
    st.markdown("## Growth Dynamics Analysis")
    
    if not data_dict:
        return
    
    # Control panel
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Cumulative Growth", "Rolling Performance", "Annual Returns", "Risk-Adjusted Metrics"]
        )
    
    with col2:
        window_size = st.slider("Rolling Window (months)", 1, 24, 12)
    
    # Prepare data
    growth_data = {}
    
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets'].copy()
            
            if analysis_type == "Cumulative Growth":
                # Calculate cumulative growth from base 100
                growth = 100 * assets / assets.iloc[0]
            
            elif analysis_type == "Rolling Performance":
                # Calculate rolling returns
                returns = assets.pct_change()
                growth = returns.rolling(window=window_size).mean() * 100
            
            elif analysis_type == "Annual Returns":
                # Calculate annualized returns
                returns = assets.pct_change()
                growth = returns.rolling(window=12).mean() * 12 * 100
            
            elif analysis_type == "Risk-Adjusted Metrics":
                # Calculate Sharpe ratio
                returns = assets.pct_change()
                rolling_sharpe = returns.rolling(window=window_size).mean() / returns.rolling(window=window_size).std()
                growth = rolling_sharpe * np.sqrt(12)  # Annualized Sharpe
            
            growth_data[category] = growth
    
    if not growth_data:
        return
    
    # Create visualization
    fig = go.Figure()
    
    # Professional color palette
    professional_colors = ['#2c3e50', '#3498db', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, (category, growth_series) in enumerate(growth_data.items()):
        color = data_dict[category]['color'] if 'color' in data_dict[category] else professional_colors[idx % len(professional_colors)]
        
        fig.add_trace(go.Scatter(
            x=growth_series.index,
            y=growth_series.iloc[:, 0] if isinstance(growth_series, pd.DataFrame) else growth_series,
            name=category,
            mode='lines',
            line=dict(width=2, color=color),
            hovertemplate='%{x|%b %Y}<br>' + f'{category}: %{{y:.2f}}<extra></extra>'
        ))
    
    # Update layout for professional appearance
    fig.update_layout(
        title=f"{analysis_type} Analysis",
        xaxis_title="Date",
        yaxis_title=analysis_type,
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("### Statistical Summary")
    
    stats_data = []
    for category, growth_series in growth_data.items():
        series = growth_series.iloc[:, 0] if isinstance(growth_series, pd.DataFrame) else growth_series
        series = series.dropna()
        
        if len(series) > 1:
            stats_data.append({
                'Category': category,
                'Mean': f"{series.mean():.3f}",
                'Std Dev': f"{series.std():.3f}",
                'Min': f"{series.min():.3f}",
                'Max': f"{series.max():.3f}",
                'Last Value': f"{series.iloc[-1]:.3f}"
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def create_flow_analysis(data_dict, frequency):
    """Create professional flow analysis"""
    st.markdown("## Flow Dynamics Analysis")
    
    if not data_dict:
        return
    
    # Inflow vs Outflow analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Inflow Analysis")
        
        # Aggregate inflows
        inflow_data = []
        dates = None
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                inflows = flows[flows.iloc[:, 0] > 0]
                
                if not inflows.empty:
                    if dates is None:
                        dates = flows.index
                    
                    inflow_df = pd.DataFrame({
                        'Date': inflows.index,
                        'Value': inflows.iloc[:, 0],
                        'Category': category
                    })
                    inflow_data.append(inflow_df)
        
        if inflow_data:
            # Create bar chart for latest inflows
            latest_inflows = []
            for data in data_dict.values():
                if 'flows' in data and not data['flows'].empty:
                    latest_flow = data['flows'].iloc[-1, 0]
                    if latest_flow > 0:
                        latest_inflows.append(latest_flow)
            
            if latest_inflows:
                fig_in = go.Figure()
                fig_in.add_trace(go.Bar(
                    x=list(data_dict.keys())[:len(latest_inflows)],
                    y=latest_inflows,
                    marker_color=['#27ae60' if x > 0 else '#e74c3c' for x in latest_inflows],
                    opacity=0.8
                ))
                
                fig_in.update_layout(
                    title=f"Latest {frequency.capitalize()} Inflows",
                    xaxis_title="Category",
                    yaxis_title="Inflows (Millions USD)",
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig_in, use_container_width=True)
    
    with col2:
        st.markdown("### Outflow Analysis")
        
        # Aggregate outflows
        outflow_data = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                outflows = flows[flows.iloc[:, 0] < 0].abs()
                
                if not outflows.empty:
                    outflow_df = pd.DataFrame({
                        'Date': outflows.index,
                        'Value': outflows.iloc[:, 0],
                        'Category': category
                    })
                    outflow_data.append(outflow_df)
        
        if outflow_data:
            # Create bar chart for latest outflows
            latest_outflows = []
            for data in data_dict.values():
                if 'flows' in data and not data['flows'].empty:
                    latest_flow = data['flows'].iloc[-1, 0]
                    if latest_flow < 0:
                        latest_outflows.append(abs(latest_flow))
            
            if latest_outflows:
                fig_out = go.Figure()
                fig_out.add_trace(go.Bar(
                    x=list(data_dict.keys())[:len(latest_outflows)],
                    y=latest_outflows,
                    marker_color='#e74c3c',
                    opacity=0.8
                ))
                
                fig_out.update_layout(
                    title=f"Latest {frequency.capitalize()} Outflows",
                    xaxis_title="Category",
                    yaxis_title="Outflows (Millions USD)",
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig_out, use_container_width=True)
    
    # Net flow analysis
    st.markdown("### Net Flow Analysis")
    
    # Calculate net flows over time
    if dates is not None:
        net_flows = pd.DataFrame(index=dates)
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                net_flows[category] = data['flows'].iloc[:, 0]
        
        if not net_flows.empty:
            fig_net = go.Figure()
            
            for column in net_flows.columns:
                fig_net.add_trace(go.Scatter(
                    x=net_flows.index,
                    y=net_flows[column],
                    name=column,
                    mode='lines',
                    line=dict(width=1),
                    hovertemplate='%{x|%b %Y}<br>' + f'{column}: $%{{y:,.0f}}M<extra></extra>'
                ))
            
            fig_net.update_layout(
                title="Net Flows Over Time",
                xaxis_title="Date",
                yaxis_title="Net Flow (Millions USD)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_net, use_container_width=True)

def create_composition_analysis(data_dict):
    """Create professional composition analysis"""
    st.markdown("## Composition Analysis")
    
    if not data_dict:
        return
    
    # Time selection
    dates = list(data_dict.values())[0]['assets'].index if data_dict else []
    
    if dates:
        selected_date = st.selectbox(
            "Select date for composition",
            options=[d.strftime('%Y-%m') for d in dates],
            index=len(dates) - 1
        )
        
        # Convert selected date back to datetime
        selected_date_dt = pd.to_datetime(selected_date)
        
        # Asset composition
        asset_values = {}
        flow_values = {}
        
        for category, data in data_dict.items():
            if 'assets' in data and selected_date_dt in data['assets'].index:
                asset_values[category] = data['assets'].loc[selected_date_dt].iloc[0]
            
            if 'flows' in data and not data['flows'].empty:
                # Get latest flow
                latest_flow = data['flows'].iloc[-1, 0]
                flow_values[category] = latest_flow
        
        col1, col2 = st.columns(2)
        
        with col1:
            if asset_values:
                st.markdown(f"### Asset Composition ({selected_date})")
                
                fig_assets = go.Figure(data=[go.Pie(
                    labels=list(asset_values.keys()),
                    values=list(asset_values.values()),
                    hole=0.3,
                    marker_colors=[data_dict[cat]['color'] for cat in asset_values.keys()],
                    textinfo='label+percent',
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                )])
                
                fig_assets.update_layout(
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_assets, use_container_width=True)
        
        with col2:
            if flow_values:
                st.markdown("### Flow Composition (Latest)")
                
                # Separate positive and negative flows
                positive_flows = {k: v for k, v in flow_values.items() if v > 0}
                negative_flows = {k: abs(v) for k, v in flow_values.items() if v < 0}
                
                if positive_flows:
                    st.markdown("**Inflows:**")
                    fig_inflows = go.Figure(data=[go.Pie(
                        labels=list(positive_flows.keys()),
                        values=list(positive_flows.values()),
                        hole=0.3,
                        marker_colors=[data_dict[cat]['color'] for cat in positive_flows.keys()],
                        hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                    )])
                    
                    fig_inflows.update_layout(
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_inflows, use_container_width=True)
                
                if negative_flows:
                    st.markdown("**Outflows:**")
                    fig_outflows = go.Figure(data=[go.Pie(
                        labels=list(negative_flows.keys()),
                        values=list(negative_flows.values()),
                        hole=0.3,
                        marker_colors=[data_dict[cat]['color'] for cat in negative_flows.keys()],
                        hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                    )])
                    
                    fig_outflows.update_layout(
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_outflows, use_container_width=True)

def create_statistical_analysis(data_dict):
    """Create professional statistical analysis"""
    st.markdown("## Statistical Analysis")
    
    if not data_dict:
        return
    
    # Create tabs for different statistical analyses
    tab1, tab2, tab3 = st.tabs(["Descriptive Statistics", "Time Series Analysis", "Risk Analysis"])
    
    with tab1:
        st.markdown("### Descriptive Statistics")
        
        stats_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].iloc[:, 0].dropna()
                
                if len(flows) > 1:
                    stats_data.append({
                        'Category': category,
                        'Observations': len(flows),
                        'Mean (M$)': f"{flows.mean():,.1f}",
                        'Std Dev (M$)': f"{flows.std():,.1f}",
                        'Minimum (M$)': f"{flows.min():,.1f}",
                        'Maximum (M$)': f"{flows.max():,.1f}",
                        'Skewness': f"{flows.skew():.3f}",
                        'Kurtosis': f"{flows.kurtosis():.3f}"
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("### Time Series Properties")
        
        selected_category = st.selectbox(
            "Select category for analysis",
            list(data_dict.keys()),
            key="ts_category"
        )
        
        if selected_category in data_dict and 'flows' in data_dict[selected_category]:
            flows = data_dict[selected_category]['flows'].iloc[:, 0].dropna()
            
            if len(flows) >= 12:
                col1, col2 = st.columns(2)
                
                with col1:
                    # ADF Test for stationarity
                    try:
                        adf_result = adfuller(flows)
                        st.markdown("**Stationarity Test (ADF):**")
                        st.write(f"Test Statistic: {adf_result[0]:.4f}")
                        st.write(f"p-value: {adf_result[1]:.4f}")
                        st.write(f"Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")
                    except:
                        st.write("Could not perform ADF test")
                
                with col2:
                    # Autocorrelation
                    try:
                        autocorr = flows.autocorr(lag=1)
                        st.markdown("**Autocorrelation:**")
                        st.write(f"Lag-1: {autocorr:.4f}")
                        st.write(f"Lag-12: {flows.autocorr(lag=12):.4f}" if len(flows) > 12 else "Insufficient data")
                    except:
                        st.write("Could not calculate autocorrelation")
    
    with tab3:
        st.markdown("### Risk Analysis")
        
        risk_data = []
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].iloc[:, 0].dropna()
                
                if len(flows) >= 12:
                    # Calculate risk metrics
                    returns = flows.pct_change().dropna()
                    
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(12)  # Annualized
                        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() != 0 else 0
                        
                        # VaR calculation
                        var_95 = np.percentile(returns, 5)
                        
                        risk_data.append({
                            'Category': category,
                            'Annual Volatility': f"{volatility:.2%}",
                            'Sharpe Ratio': f"{sharpe:.3f}",
                            'VaR (95%)': f"{var_95:.2%}",
                            'Max Drawdown': f"{((1 + returns).cumprod().expanding().max() - (1 + returns).cumprod()).max():.2%}"
                        })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, height=400)

def create_advanced_data_table(data_dict, frequency):
    """Create advanced data table with export capabilities"""
    st.markdown("## Advanced Data Table")
    
    if not data_dict:
        return
    
    # Create combined dataframe
    combined_data = []
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            df = data['flows'].copy()
            df.columns = [f'{category}_Flow']
            combined_data.append(df)
    
    if combined_data:
        # Combine all data
        all_data = pd.concat(combined_data, axis=1)
        
        # Format for display
        display_df = all_data.copy()
        display_df.index.name = 'Date'
        display_df = display_df.reset_index()
        
        # Add formatted columns
        for col in display_df.columns:
            if col != 'Date' and display_df[col].dtype in ['float64', 'int64']:
                display_df[f'{col}_Formatted'] = display_df[col].apply(
                    lambda x: f"${x:,.0f}M" if pd.notnull(x) else ""
                )
        
        # Display table with configurable options
        st.markdown("### Data Preview")
        
        # Row limit selector
        row_limit = st.slider("Rows to display", 10, 100, 20)
        
        st.dataframe(
            display_df.head(row_limit),
            use_container_width=True,
            height=400
        )
        
        # Export options
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = all_data.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fund_flows_{frequency}_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary statistics download
            summary_stats = []
            for category, data in data_dict.items():
                if 'flows' in data and not data['flows'].empty:
                    flows = data['flows'].iloc[:, 0]
                    summary_stats.append({
                        'Category': category,
                        'Mean': flows.mean(),
                        'Std Dev': flows.std(),
                        'Min': flows.min(),
                        'Max': flows.max(),
                        'Latest': flows.iloc[-1] if len(flows) > 0 else None
                    })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=summary_csv,
                    file_name=f"summary_stats_{datetime.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

def main():
    """Main application function"""
    
    # Sidebar with clean design
    with st.sidebar:
        st.markdown("### Data Configuration")
        
        # Frequency selection - user can freely select
        frequency = st.radio(
            "Data Frequency",
            ['weekly', 'monthly'],
            index=1,
            help="Select weekly or monthly frequency"
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
        
        # Filter series based on selection
        selected_series = {k: FRED_SERIES[k] for k in selected_categories if k in FRED_SERIES}
        
        st.markdown("---")
        
        # Data status
        st.markdown("### Data Status")
        st.markdown(f"""
        - Frequency: **{frequency}**
        - Period: **{start_date}** to present
        - Categories: **{len(selected_series)}** selected
        - Source: **Federal Reserve (FRED)**
        """)
    
    # Load data
    if not selected_series:
        st.warning("Please select at least one fund category from the sidebar.")
        return
    
    with st.spinner(f"Loading {frequency} data from FRED..."):
        data_dict = load_all_data(selected_series, str(start_date), frequency)
    
    if not data_dict:
        st.error("No data available. Please check your selections and try again.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Growth Dynamics",
        "Flow Dynamics",
        "Composition",
        "Statistical Analysis",
        "Data Table"
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
        create_statistical_analysis(data_dict)
    
    with tab6:
        create_advanced_data_table(data_dict, frequency)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Institutional Fund Flow Analytics</strong></p>
        <p>Data Source: Federal Reserve Economic Data (FRED) | Frequency: {frequency} | Period: {start_date} to {end_date}</p>
        <p>All figures in millions of USD | Professional analytics platform</p>
    </div>
    """.format(
        frequency=frequency,
        start_date=start_date.strftime('%Y-%m'),
        end_date=datetime.today().strftime('%Y-%m')
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
