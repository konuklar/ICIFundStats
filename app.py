import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_datareader.data as web
import warnings
from scipy import stats
import calendar
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FRED Mutual Fund Flows - Institutional Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4B5563;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-change {
        font-size: 0.9rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 5px;
    }
    .positive {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .negative {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .section-header {
        font-size: 1.6rem;
        color: #1E3A8A;
        font-weight: 600;
        margin: 2.5rem 0 1.2rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3B82F6;
    }
    .data-table-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.5rem;
        background: #F9FAFB;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        border-radius: 8px;
        background: white;
        border: 1px solid #E5E7EB;
        font-weight: 500;
        color: #4B5563;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .institution-logo {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6B7280;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with institutional styling
st.markdown('<h1 class="main-header">🏦 FRED Mutual Fund Flows - Institutional Analysis Suite</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive U.S. Mutual Fund & ETF Flow Analysis | Federal Reserve Economic Data (FRED)</p>', unsafe_allow_html=True)

# FRED Series IDs for mutual fund flows
FRED_SERIES = {
    'Total Mutual Fund Assets': {
        'monthly': 'TOTMFS',
        'weekly': 'H8B3092NCBA',
        'description': 'Total Mutual Fund Assets'
    },
    'Money Market Funds': {
        'monthly': 'MMMFFAQ027S',
        'weekly': 'H6BMMTNA',
        'description': 'Money Market Mutual Fund Assets'
    },
    'Equity Funds': {
        'monthly': 'TOTCI',
        'weekly': 'H8B3053NCBA',
        'description': 'Total Equity Mutual Fund Assets'
    },
    'Bond Funds': {
        'monthly': 'TBCI',
        'weekly': 'H8B3094NCBA',
        'description': 'Total Bond/Income Mutual Fund Assets'
    },
    'Hybrid Funds': {
        'monthly': 'H8B3095NCBA',
        'weekly': 'H8B3095NCBA',
        'description': 'Hybrid Mutual Fund Assets'
    },
    'Municipal Bond Funds': {
        'monthly': 'MBCI',
        'weekly': None,
        'description': 'Municipal Bond Fund Assets'
    }
}

@st.cache_data(ttl=3600, show_spinner="Fetching latest FRED data...")
def fetch_fred_data(series_dict, start_date='2007-01-01', frequency='monthly'):
    """Fetch data from FRED for multiple series"""
    data = {}
    failed_series = []
    
    for category, info in series_dict.items():
        series_id = info.get(frequency)
        if series_id:
            try:
                # Fetch data from FRED
                df = web.DataReader(series_id, 'fred', start=start_date, end=datetime.today())
                if not df.empty:
                    # Calculate flows (monthly/weekly changes)
                    df_flows = df.diff()
                    df_flows.columns = [f'{category}_Flow']
                    
                    # Store both levels and flows
                    data[category] = {
                        'assets': df,
                        'flows': df_flows,
                        'description': info['description']
                    }
                    
                    st.sidebar.success(f"✓ {category}")
                else:
                    failed_series.append(category)
            except Exception as e:
                failed_series.append(category)
                st.sidebar.error(f"✗ {category}: {str(e)[:50]}...")
        else:
            failed_series.append(category)
    
    if failed_series:
        st.sidebar.warning(f"Could not load: {', '.join(failed_series)}")
    
    return data

def create_institutional_metrics(data_dict, frequency):
    """Create institutional metrics dashboard"""
    if not data_dict:
        return
    
    st.markdown("## 📊 Executive Dashboard")
    
    # Create metrics row
    cols = st.columns(5)
    
    metrics_data = []
    for idx, (category, data) in enumerate(list(data_dict.items())[:5]):
        with cols[idx]:
            if 'flows' in data and not data['flows'].empty:
                latest_flow = data['flows'].iloc[-1, 0]
                avg_flow = data['flows'].mean().iloc[0]
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{category}</div>
                    <div class='metric-value'>${latest_flow:,.0f}M</div>
                    <div class='metric-change {'positive' if latest_flow > 0 else 'negative'}">
                        {'+' if latest_flow > 0 else ''}{latest_flow:,.0f}M {frequency}
                    </div>
                    <div style='font-size: 0.8rem; color: #6B7280; margin-top: 8px;'>
                        Avg: ${avg_flow:,.0f}M
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                metrics_data.append({
                    'Category': category,
                    f'Latest {frequency.capitalize()} Flow': f"${latest_flow:,.0f}M",
                    'Direction': 'Inflow' if latest_flow > 0 else 'Outflow',
                    '12M Avg': f"${avg_flow:,.0f}M"
                })
    
    return pd.DataFrame(metrics_data)

def create_normalized_growth_charts(data_dict):
    """Create normalized growth charts without overlap"""
    if not data_dict:
        return
    
    st.markdown('<div class="section-header">🌱 Normalized Growth Dynamics</div>', unsafe_allow_html=True)
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    with col1:
        norm_method = st.selectbox(
            "Normalization Method",
            ["Cumulative Growth Index", "Z-Score", "Percentage of Peak", "Rolling Average"],
            key="norm_method"
        )
    with col2:
        window = st.slider("Smoothing Window", 1, 12, 3, key="smoothing_window")
    with col3:
        show_separate = st.checkbox("Show Separate Charts", True)
    
    # Prepare data for normalization
    all_dates = None
    normalized_data = []
    
    for category, data in data_dict.items():
        if 'assets' in data and not data['assets'].empty:
            # Get asset data
            assets = data['assets'].copy()
            assets.columns = [category]
            
            if all_dates is None:
                all_dates = assets.index
            
            # Normalize based on selected method
            if norm_method == "Cumulative Growth Index":
                # Base = 100 at start
                normalized = 100 * assets / assets.iloc[0]
            elif norm_method == "Z-Score":
                normalized = (assets - assets.mean()) / assets.std()
            elif norm_method == "Percentage of Peak":
                normalized = 100 * assets / assets.max()
            elif norm_method == "Rolling Average":
                normalized = assets.rolling(window=window).mean()
            
            normalized_data.append(normalized)
    
    if not normalized_data:
        return
    
    # Combine all normalized data
    combined_df = pd.concat(normalized_data, axis=1)
    
    if show_separate and len(combined_df.columns) > 2:
        # Create separate subplots for clarity
        n_cols = 2
        n_rows = (len(combined_df.columns) + 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=combined_df.columns.tolist(),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, column in enumerate(combined_df.columns):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            # Apply smoothing
            if window > 1:
                y_data = combined_df[column].rolling(window=window).mean()
            else:
                y_data = combined_df[column]
            
            fig.add_trace(
                go.Scatter(
                    x=combined_df.index,
                    y=y_data,
                    name=column,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=3),
                    fill='tozeroy' if norm_method == "Cumulative Growth Index" else None,
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.2])}',
                    hovertemplate=f'%{{x|%b %Y}}<br>{column}: %{{y:.2f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=300 * n_rows,
            showlegend=False,
            title_text=f"Normalized Growth by Category ({norm_method})",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update y-axis titles
        for i in range(1, n_rows * n_cols + 1):
            fig.update_yaxes(title_text="Normalized Value", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        
    else:
        # Single chart with clear differentiation
        fig = go.Figure()
        
        colors = px.colors.qualitative.Bold
        line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        
        for i, column in enumerate(combined_df.columns):
            # Apply smoothing
            if window > 1:
                y_data = combined_df[column].rolling(window=window).mean()
            else:
                y_data = combined_df[column]
            
            fig.add_trace(go.Scatter(
                x=combined_df.index,
                y=y_data,
                name=column,
                mode='lines',
                line=dict(
                    color=colors[i % len(colors)],
                    width=3,
                    dash=line_styles[i % len(line_styles)]
                ),
                hovertemplate=f'%{{x|%b %Y}}<br>{column}: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Normalized Growth Trends ({norm_method})",
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.9)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif")
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add statistics table
    st.markdown("#### 📈 Growth Statistics")
    
    stats_data = []
    for column in combined_df.columns:
        if len(combined_df[column]) > 1:
            start_val = combined_df[column].iloc[0]
            end_val = combined_df[column].iloc[-1]
            growth_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
            
            stats_data.append({
                'Category': column,
                'Start Value': f"{start_val:.2f}",
                'End Value': f"{end_val:.2f}",
                'Total Growth %': f"{growth_pct:+.2f}%",
                'Volatility': f"{combined_df[column].std():.3f}"
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def create_flow_analysis(data_dict, frequency):
    """Create separate inflow/outflow analysis"""
    if not data_dict:
        return
    
    st.markdown('<div class="section-header">📊 Flow Direction Analysis</div>', unsafe_allow_html=True)
    
    # Separate inflow and outflow charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Monthly Inflows")
        
        # Prepare inflow data
        inflow_data = []
        dates = None
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                if dates is None:
                    dates = flows.index
                
                # Get positive flows only
                positive_flows = flows[flows.iloc[:, 0] > 0].copy()
                positive_flows.columns = [category]
                inflow_data.append(positive_flows)
        
        if inflow_data and dates is not None:
            # Combine all inflow data
            inflows_df = pd.concat(inflow_data, axis=1).fillna(0)
            inflows_df = inflows_df.reindex(dates).fillna(0)
            
            # Create stacked area chart
            fig_inflows = go.Figure()
            
            colors = px.colors.qualitative.Pastel
            for i, column in enumerate(inflows_df.columns):
                fig_inflows.add_trace(go.Scatter(
                    x=inflows_df.index,
                    y=inflows_df[column],
                    name=column,
                    mode='lines',
                    stackgroup='one',
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)],
                    hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                ))
            
            fig_inflows.update_layout(
                title=f"Monthly Inflows by Category",
                xaxis_title="Date",
                yaxis_title="Inflows (Millions USD)",
                height=350,
                showlegend=True,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_inflows, use_container_width=True)
            
            # Inflow statistics
            total_inflows = inflows_df.sum().sum()
            st.metric("Total Period Inflows", f"${total_inflows:,.0f}M")
    
    with col2:
        st.markdown("##### 📉 Monthly Outflows")
        
        # Prepare outflow data
        outflow_data = []
        
        for category, data in data_dict.items():
            if 'flows' in data and not data['flows'].empty:
                flows = data['flows'].copy()
                
                # Get negative flows only (convert to positive for display)
                negative_flows = flows[flows.iloc[:, 0] < 0].copy()
                negative_flows.iloc[:, 0] = negative_flows.iloc[:, 0].abs()
                negative_flows.columns = [category]
                outflow_data.append(negative_flows)
        
        if outflow_data and dates is not None:
            # Combine all outflow data
            outflows_df = pd.concat(outflow_data, axis=1).fillna(0)
            outflows_df = outflows_df.reindex(dates).fillna(0)
            
            # Create stacked area chart
            fig_outflows = go.Figure()
            
            colors = px.colors.qualitative.Pastel2
            for i, column in enumerate(outflows_df.columns):
                fig_outflows.add_trace(go.Scatter(
                    x=outflows_df.index,
                    y=outflows_df[column],
                    name=column,
                    mode='lines',
                    stackgroup='one',
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)],
                    hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                ))
            
            fig_outflows.update_layout(
                title=f"Monthly Outflows by Category",
                xaxis_title="Date",
                yaxis_title="Outflows (Millions USD)",
                height=350,
                showlegend=True,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_outflows, use_container_width=True)
            
            # Outflow statistics
            total_outflows = outflows_df.sum().sum()
            st.metric("Total Period Outflows", f"${total_outflows:,.0f}M")
    
    # Net flow analysis
    st.markdown("##### ⚖️ Net Flow Analysis")
    
    net_flow_data = []
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flows = data['flows'].copy()
            flows.columns = [category]
            net_flow_data.append(flows)
    
    if net_flow_data:
        net_flows_df = pd.concat(net_flow_data, axis=1)
        
        # Calculate cumulative net flows
        cumulative_net = net_flows_df.cumsum()
        
        fig_net = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Monthly Net Flows', 'Cumulative Net Flows'],
            vertical_spacing=0.15
        )
        
        # Monthly net flows
        for i, column in enumerate(net_flows_df.columns):
            fig_net.add_trace(
                go.Bar(
                    x=net_flows_df.index,
                    y=net_flows_df[column],
                    name=column,
                    marker_color=['#10B981' if x > 0 else '#EF4444' for x in net_flows_df[column]],
                    opacity=0.7,
                    hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Cumulative net flows
        for i, column in enumerate(cumulative_net.columns):
            fig_net.add_trace(
                go.Scatter(
                    x=cumulative_net.index,
                    y=cumulative_net[column],
                    name=f"{column} Cumulative",
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate=f'%{{x|%b %Y}}<br>{column}: $%{{y:,.0f}}M<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig_net.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor='white'
        )
        
        fig_net.update_yaxes(title_text="Net Flow (M USD)", row=1, col=1)
        fig_net.update_yaxes(title_text="Cumulative Flow (M USD)", row=2, col=1)
        
        st.plotly_chart(fig_net, use_container_width=True)

def create_dynamic_composition(data_dict):
    """Create dynamic pie charts for composition analysis"""
    if not data_dict:
        return
    
    st.markdown('<div class="section-header">🥧 Dynamic Composition Analysis</div>', unsafe_allow_html=True)
    
    # Time slider for dynamic visualization
    if 'assets' in list(data_dict.values())[0]:
        assets_data = list(data_dict.values())[0]['assets']
        if not assets_data.empty:
            time_index = st.slider(
                "Select time period for composition",
                0, len(assets_data)-1,
                len(assets_data)-1,
                format=lambda x: f"{assets_data.index[int(x)].strftime('%B %Y')}"
            )
            
            selected_date = assets_data.index[time_index]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Asset composition
                asset_values = {}
                for category, data in data_dict.items():
                    if 'assets' in data and selected_date in data['assets'].index:
                        asset_values[category] = data['assets'].loc[selected_date].iloc[0]
                
                if asset_values:
                    fig_assets = go.Figure(data=[go.Pie(
                        labels=list(asset_values.keys()),
                        values=list(asset_values.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set3),
                        textinfo='label+percent',
                        hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                    )])
                    
                    fig_assets.update_layout(
                        title=f"Asset Composition - {selected_date.strftime('%b %Y')}",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_assets, use_container_width=True)
            
            with col2:
                # Flow composition (latest month)
                flow_values = {}
                for category, data in data_dict.items():
                    if 'flows' in data and not data['flows'].empty:
                        latest_flow = data['flows'].iloc[-1, 0]
                        if abs(latest_flow) > 0:
                            flow_values[category] = abs(latest_flow)
                
                if flow_values:
                    fig_flows = go.Figure(data=[go.Pie(
                        labels=list(flow_values.keys()),
                        values=list(flow_values.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Pastel),
                        textinfo='label+percent',
                        hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                    )])
                    
                    fig_flows.update_layout(
                        title=f"Latest {list(data_dict.values())[0]['flows'].index[-1].strftime('%b %Y')} Flow Composition",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_flows, use_container_width=True)
            
            with col3:
                # Year-to-date flows
                ytd_values = {}
                current_year = selected_date.year
                
                for category, data in data_dict.items():
                    if 'flows' in data and not data['flows'].empty:
                        ytd_flow = data['flows'][data['flows'].index.year == current_year].sum().iloc[0]
                        if abs(ytd_flow) > 0:
                            ytd_values[category] = abs(ytd_flow)
                
                if ytd_values:
                    fig_ytd = go.Figure(data=[go.Pie(
                        labels=list(ytd_values.keys()),
                        values=list(ytd_values.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Bold),
                        textinfo='label+percent',
                        hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}<extra></extra>"
                    )])
                    
                    fig_ytd.update_layout(
                        title=f"Year-to-Date Flow Composition ({current_year})",
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_ytd, use_container_width=True)

def create_data_table(data_dict, frequency):
    """Create comprehensive data table"""
    if not data_dict:
        return
    
    st.markdown('<div class="section-header">📋 Detailed Data Table</div>', unsafe_allow_html=True)
    
    # Create combined dataframe
    flow_data = []
    asset_data = []
    
    for category, data in data_dict.items():
        if 'flows' in data and not data['flows'].empty:
            flows = data['flows'].copy()
            flows.columns = [f'{category}_Flow']
            flow_data.append(flows)
        
        if 'assets' in data and not data['assets'].empty:
            assets = data['assets'].copy()
            assets.columns = [f'{category}_Assets']
            asset_data.append(assets)
    
    if flow_data:
        combined_flows = pd.concat(flow_data, axis=1)
        combined_assets = pd.concat(asset_data, axis=1) if asset_data else pd.DataFrame()
        
        # Merge flows and assets
        if not combined_assets.empty:
            combined_df = pd.concat([combined_flows, combined_assets], axis=1)
        else:
            combined_df = combined_flows
        
        # Format for display
        display_df = combined_df.copy()
        display_df.index.name = 'Date'
        display_df = display_df.reset_index()
        
        # Format numeric columns
        for col in display_df.columns:
            if col != 'Date' and display_df[col].dtype in ['float64', 'int64']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x:,.0f}M" if pd.notnull(x) else ""
                )
        
        # Display table
        st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download options
        csv = combined_df.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"fred_mutual_fund_flows_{frequency}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="institution-logo">
            <h3 style="color: #1E3A8A; margin: 0;">Federal Reserve</h3>
            <h4 style="color: #3B82F6; margin: 0;">Economic Data (FRED)</h4>
            <p style="color: #6B7280; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                Mutual Fund Flows Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data frequency selection
        st.markdown("### 📅 Data Frequency")
        frequency = st.radio(
            "Select data frequency",
            ['monthly', 'weekly'],
            index=0,
            horizontal=True
        )
        
        # Date range selection
        st.markdown("### ⏳ Date Range")
        start_date = st.date_input(
            "Start Date",
            value=datetime(2007, 1, 1),
            min_value=datetime(1990, 1, 1),
            max_value=datetime.today()
        )
        
        # Fund categories selection
        st.markdown("### 🏦 Fund Categories")
        all_categories = list(FRED_SERIES.keys())
        selected_categories = st.multiselect(
            "Select categories to analyze",
            all_categories,
            default=all_categories[:4]
        )
        
        # Filter FRED series based on selection
        filtered_series = {k: FRED_SERIES[k] for k in selected_categories if k in FRED_SERIES}
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard provides institutional-grade analysis of U.S. mutual fund flows using official Federal Reserve Economic Data (FRED).
        
        **Data Source:** Federal Reserve Bank of St. Louis
        **Frequency:** Monthly & Weekly
        **Units:** Millions of USD
        **Last Updated:** Real-time via FRED API
        """)
    
    # Load data
    if not filtered_series:
        st.warning("Please select at least one fund category from the sidebar.")
        return
    
    with st.spinner(f"Fetching {frequency} data from FRED..."):
        data_dict = fetch_fred_data(filtered_series, start_date=str(start_date), frequency=frequency)
    
    if not data_dict:
        st.error("Could not load data from FRED. Please check your selections and try again.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary",
        "📈 Growth Dynamics",
        "📉 Flow Analysis",
        "🥧 Composition",
        "📋 Data Explorer"
    ])
    
    with tab1:
        # Institutional metrics
        metrics_df = create_institutional_metrics(data_dict, frequency)
        
        # Performance summary
        st.markdown('<div class="section-header">📈 Performance Summary</div>', unsafe_allow_html=True)
        
        if metrics_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                # Overall statistics
                if data_dict:
                    total_latest_flow = sum([
                        data['flows'].iloc[-1, 0] if 'flows' in data and not data['flows'].empty else 0
                        for data in data_dict.values()
                    ])
                    
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-label'>Total Latest Flow</div>
                        <div class='metric-value'>${:,.0f}M</div>
                        <div class='metric-change {}'>
                            {}${:,.0f}M
                        </div>
                    </div>
                    """.format(
                        abs(total_latest_flow),
                        'positive' if total_latest_flow > 0 else 'negative',
                        '+' if total_latest_flow > 0 else '',
                        total_latest_flow
                    ), unsafe_allow_html=True)
    
    with tab2:
        # Normalized growth charts
        create_normalized_growth_charts(data_dict)
    
    with tab3:
        # Flow analysis
        create_flow_analysis(data_dict, frequency)
    
    with tab4:
        # Composition analysis
        create_dynamic_composition(data_dict)
    
    with tab5:
        # Data table
        create_data_table(data_dict, frequency)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>FRED Mutual Fund Flows - Institutional Dashboard</strong></p>
        <p>Data Source: Federal Reserve Economic Data (FRED) | Federal Reserve Bank of St. Louis</p>
        <p>All figures in millions of U.S. dollars | Negative values indicate net outflows</p>
        <p>Dashboard v3.0 | Generated on {}</p>
    </div>
    """.format(datetime.today().strftime('%B %d, %Y')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
