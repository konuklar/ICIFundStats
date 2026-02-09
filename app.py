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
import calendar
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ICI Mutual Fund Flows - Institutional Dashboard",
    page_icon="🏦",
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
    .data-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .positive-flow {
        color: #10B981;
        font-weight: 600;
    }
    .negative-flow {
        color: #EF4444;
        font-weight: 600;
    }
    .year-highlight {
        background-color: #EFF6FF;
        font-weight: 600;
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
    .download-btn {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Title with institutional styling
st.markdown('<h1 class="main-header">🏦 ICI Mutual Fund Flows - Institutional Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive Monthly Analysis of Net New Cash Flows by Investment Class | 2007 - Present</p>', unsafe_allow_html=True)

# Enhanced sample data generation
@st.cache_data
def generate_enhanced_sample_data(start_date='2007-01-01', end_date=None):
    """Generate realistic sample data with seasonal patterns and correlations"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    n = len(dates)
    np.random.seed(42)
    
    # Time index for trend calculations
    t = np.arange(n)
    
    # Generate base components with realistic correlations
    # Common market factor affecting all categories
    market_factor = 5000 * np.sin(2 * np.pi * t / 60)  # 5-year market cycle
    market_factor += 2000 * np.sin(2 * np.pi * t / 12)  # Annual seasonality
    market_factor += np.random.normal(0, 3000, n)  # Random noise
    
    # Equity funds - highest volatility, positive long-term trend
    equity_trend = 150 * t  # Long-term growth
    equity_seasonal = 4000 * np.sin(2 * np.pi * t / 12 + np.pi/4)  # Strong seasonality
    equity = 8000 + equity_trend + 1.5 * market_factor + equity_seasonal
    
    # Bond funds - moderate growth, negatively correlated with equity during crises
    bond_trend = 80 * t
    bond_seasonal = 2000 * np.sin(2 * np.pi * t / 12)
    bond = 5000 + bond_trend + 0.8 * market_factor + bond_seasonal
    
    # Money Market - flight to safety during crises
    mm_trend = 40 * t
    mm_seasonal = 3000 * np.sin(2 * np.pi * t / 12 - np.pi/2)
    money_market = 10000 + mm_trend + 1.2 * market_factor + mm_seasonal
    
    # Hybrid funds - mixture of equity and bond
    hybrid = 0.6 * equity + 0.4 * bond + np.random.normal(0, 1500, n)
    
    # Add specific crisis events
    # 2008 Financial Crisis
    crisis_2008 = (t >= 20) & (t <= 25)
    equity[crisis_2008] -= 25000
    bond[crisis_2008] += 18000
    money_market[crisis_2008] += 60000
    
    # 2020 COVID Crisis
    crisis_2020 = (t >= 160) & (t <= 165)
    equity[crisis_2020] -= 35000
    bond[crisis_2020] += 22000
    money_market[crisis_2020] += 85000
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Year': dates.year,
        'Month': dates.month,
        'Month_Name': dates.strftime('%B'),
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
    
    # Calculate monthly changes
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Monthly_Change'] = df[category].pct_change() * 100
        df[f'{category}_Normalized'] = 100 * (df[category] - df[category].mean()) / df[category].std()
    
    # Add inflow/outflow indicators
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Inflow'] = df[category].clip(lower=0)
        df[f'{category}_Outflow'] = df[category].clip(upper=0).abs()
    
    # Add quarterly and annual aggregations
    df['Quarter'] = df['Date'].dt.quarter
    df['YearQuarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    
    return df

# Load data
df = generate_enhanced_sample_data()

# Sidebar controls
with st.sidebar:
    st.image("https://www.ici.org/themes/custom/ici/logo.svg", width=200)
    st.markdown("---")
    
    st.header("📊 Dashboard Controls")
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=365*5), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Categories selection
    st.subheader("Fund Categories")
    categories = st.multiselect(
        "Select categories for analysis",
        ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total'],
        default=['Equity', 'Bond', 'Money Market']
    )
    
    # Display options
    st.subheader("Display Options")
    show_cumulative = st.checkbox("Show Cumulative Data", True)
    show_monthly_changes = st.checkbox("Show Monthly Changes", True)
    show_normalized = st.checkbox("Show Normalized Views", True)
    
    # Data aggregation
    st.subheader("Data Aggregation")
    aggregation_level = st.selectbox(
        "Aggregation Level",
        ["Monthly", "Quarterly", "Annual"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Data Source")
    st.markdown("**Investment Company Institute (ICI)**")
    st.markdown("Monthly Mutual Fund Flows")
    st.markdown("All figures in millions USD")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Monthly Data Table",
    "📈 Growth Dynamics",
    "📊 Inflow/Outflow Analysis",
    "🥧 Composition Analysis",
    "📉 Performance Metrics"
])

with tab1:
    # Monthly Data Table - Enhanced with filtering and sorting
    st.markdown('<div class="section-header">📋 Monthly Mutual Fund Flows Data</div>', unsafe_allow_html=True)
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_flow = df_filtered['Total'].sum()
        st.metric("Total Net Flow", f"${total_flow:,.0f}M")
    
    with col2:
        avg_monthly = df_filtered['Total'].mean()
        st.metric("Avg Monthly Flow", f"${avg_monthly:,.0f}M")
    
    with col3:
        inflow_months = (df_filtered['Total'] > 0).sum()
        total_months = len(df_filtered)
        st.metric("Inflow Months", f"{inflow_months}/{total_months}")
    
    with col4:
        latest_date = df_filtered['Date'].iloc[-1].strftime('%B %Y')
        st.metric("Latest Data", latest_date)
    
    # Data filtering controls
    st.markdown("### Data Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        year_filter = st.multiselect(
            "Filter by Year",
            options=sorted(df_filtered['Year'].unique(), reverse=True),
            default=[]
        )
    
    with filter_col2:
        month_filter = st.multiselect(
            "Filter by Month",
            options=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x]
        )
    
    with filter_col3:
        flow_direction = st.selectbox(
            "Flow Direction",
            ["All Flows", "Inflows Only", "Outflows Only"]
        )
    
    # Apply filters
    df_display = df_filtered.copy()
    
    if year_filter:
        df_display = df_display[df_display['Year'].isin(year_filter)]
    
    if month_filter:
        df_display = df_display[df_display['Month'].isin(month_filter)]
    
    if flow_direction == "Inflows Only":
        df_display = df_display[df_display['Total'] > 0]
    elif flow_direction == "Outflows Only":
        df_display = df_display[df_display['Total'] < 0]
    
    # Data table with enhanced formatting
    st.markdown("### Monthly Flow Data")
    
    # Create display dataframe with formatted columns
    display_cols = ['Date', 'Year', 'Month_Name'] + categories
    if 'Total' not in categories:
        display_cols.append('Total')
    
    df_formatted = df_display[display_cols].copy()
    
    # Format numeric columns
    for col in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"<span class='positive-flow'>${x:,.0f}M</span>" if x > 0 
                else f"<span class='negative-flow'>${x:,.0f}M</span>" if x < 0 
                else f"${x:,.0f}M"
            )
    
    # Display table
    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    
    # Use Streamlit's data_editor for interactive features
    st.data_editor(
        df_formatted,
        use_container_width=True,
        height=500,
        column_config={
            "Date": st.column_config.DateColumn(
                "Date",
                format="YYYY-MM",
                width="small"
            ),
            "Year": st.column_config.NumberColumn(
                "Year",
                format="%d",
                width="small"
            ),
            "Month_Name": st.column_config.TextColumn(
                "Month",
                width="small"
            ),
            "Equity": st.column_config.TextColumn(
                "Equity",
                width="medium"
            ),
            "Bond": st.column_config.TextColumn(
                "Bond",
                width="medium"
            ),
            "Hybrid": st.column_config.TextColumn(
                "Hybrid",
                width="medium"
            ),
            "Money Market": st.column_config.TextColumn(
                "Money Market",
                width="medium"
            ),
            "Total": st.column_config.TextColumn(
                "Total",
                width="medium"
            )
        },
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    summary_cols = st.columns(len(categories))
    
    for idx, category in enumerate(categories):
        with summary_cols[idx]:
            if category in df_display.columns:
                stats_data = {
                    'Mean': f"${df_display[category].mean():,.0f}M",
                    'Median': f"${df_display[category].median():,.0f}M",
                    'Std Dev': f"${df_display[category].std():,.0f}M",
                    'Min': f"${df_display[category].min():,.0f}M",
                    'Max': f"${df_display[category].max():,.0f}M"
                }
                
                st.markdown(f"**{category}**")
                for stat, value in stats_data.items():
                    st.markdown(f"{stat}: {value}")
    
    # Download options
    st.markdown("### Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="ici_monthly_flows.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_display.to_excel(writer, sheet_name='Monthly Flows', index=False)
            # Add summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Total Months', 'Total Net Flow', 'Average Monthly', 
                          'Max Inflow', 'Max Outflow', 'Inflow Months', 'Outflow Months'],
                'Value': [
                    len(df_display),
                    f"${df_display['Total'].sum():,.0f}M",
                    f"${df_display['Total'].mean():,.0f}M",
                    f"${df_display['Total'].max():,.0f}M",
                    f"${abs(df_display['Total'].min()):,.0f}M",
                    (df_display['Total'] > 0).sum(),
                    (df_display['Total'] < 0).sum()
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="ici_flows_detailed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # Quick summary PDF (simulated)
        st.download_button(
            label="📄 Download Summary PDF",
            data=csv,
            file_name="ici_flows_summary.txt",
            mime="text/plain"
        )

with tab2:
    # Growth Dynamics - Fixed overlapping issues
    st.markdown('<div class="section-header">📈 Normalized Growth Trends Analysis</div>', unsafe_allow_html=True)
    
    # Control panel for growth charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        normalization_type = st.selectbox(
            "Normalization Method",
            ["Index (Base=100)", "Z-Score", "Min-Max Scaling", "Percentage of Mean"]
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Area Chart", "Bar Chart", "Combined"]
        )
    
    with col3:
        smoothing_window = st.slider("Smoothing Window (months)", 1, 12, 3)
    
    # Create non-overlapping growth charts
    st.markdown("### Cumulative Growth Index (Base = 100)")
    
    # Calculate normalized growth
    normalized_data = pd.DataFrame()
    normalized_data['Date'] = df_filtered['Date']
    
    for category in categories:
        if f'{category}_Cumulative' in df_filtered.columns:
            cumulative = df_filtered[f'{category}_Cumulative']
            
            if normalization_type == "Index (Base=100)":
                normalized_data[category] = 100 * cumulative / cumulative.iloc[0]
            elif normalization_type == "Z-Score":
                normalized_data[category] = (cumulative - cumulative.mean()) / cumulative.std()
            elif normalization_type == "Min-Max Scaling":
                normalized_data[category] = 100 * (cumulative - cumulative.min()) / (cumulative.max() - cumulative.min())
            elif normalization_type == "Percentage of Mean":
                normalized_data[category] = 100 * cumulative / cumulative.mean()
    
    # Create separate charts for better readability
    if len(categories) <= 3:
        # Use single chart with clear differentiation
        fig_growth = go.Figure()
        
        colors = px.colors.qualitative.Bold
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        
        for i, category in enumerate(categories):
            if category in normalized_data.columns:
                # Apply smoothing if selected
                if smoothing_window > 1:
                    y_data = normalized_data[category].rolling(window=smoothing_window).mean()
                else:
                    y_data = normalized_data[category]
                
                fig_growth.add_trace(go.Scatter(
                    x=normalized_data['Date'],
                    y=y_data,
                    name=category,
                    mode='lines',
                    line=dict(
                        color=colors[i % len(colors)],
                        width=3,
                        dash=line_styles[i % len(line_styles)]
                    ),
                    hovertemplate='%{x|%b %Y}<br>' +
                                f'{category}: %{{y:.1f}}<br>' +
                                '<extra></extra>'
                ))
        
        fig_growth.update_layout(
            title=f"Normalized Growth Trends ({normalization_type})",
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
    
    else:
        # Use subplots for 4+ categories
        n_cols = 2
        n_rows = (len(categories) + 1) // 2
        
        fig_growth = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=categories,
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        for i, category in enumerate(categories):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            if category in normalized_data.columns:
                # Apply smoothing if selected
                if smoothing_window > 1:
                    y_data = normalized_data[category].rolling(window=smoothing_window).mean()
                else:
                    y_data = normalized_data[category]
                
                fig_growth.add_trace(
                    go.Scatter(
                        x=normalized_data['Date'],
                        y=y_data,
                        name=category,
                        mode='lines',
                        line=dict(color=px.colors.qualitative.Set1[i % 10], width=2)
                    ),
                    row=row, col=col
                )
        
        fig_growth.update_layout(
            title=f"Normalized Growth Trends by Category ({normalization_type})",
            height=300 * n_rows,
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
    
    # Dynamic pie charts for composition over time
    st.markdown("### Dynamic Composition Analysis")
    
    # Time slider for pie chart animation
    if 'Date' in df_filtered.columns:
        time_index = st.slider(
            "Select time period for composition analysis",
            0, len(df_filtered)-1,
            len(df_filtered)-1,
            format="Month %d (%Y)"
        )
        
        selected_date = df_filtered['Date'].iloc[time_index]
        selected_data = df_filtered.iloc[time_index]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Inflow composition pie chart
            inflow_data = {}
            for category in categories:
                if f'{category}_Inflow' in selected_data:
                    inflow_val = selected_data[f'{category}_Inflow']
                    if inflow_val > 0:
                        inflow_data[category] = inflow_val
            
            if inflow_data:
                fig_inflow_pie = go.Figure(data=[go.Pie(
                    labels=list(inflow_data.keys()),
                    values=list(inflow_data.values()),
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    hovertemplate="%{label}<br>$%{value:.0f}M<br>%{percent}"
                )])
                
                fig_inflow_pie.update_layout(
                    title=f"Inflow Composition - {selected_date.strftime('%b %Y')}",
                    height=350
                )
                st.plotly_chart(fig_inflow_pie, use_container_width=True)
        
        with col2:
            # Outflow composition pie chart
            outflow_data = {}
            for category in categories:
                if f'{category}_Outflow' in selected_data:
                    outflow_val = selected_data[f'{category}_Outflow']
                    if outflow_val > 0:
                        outflow_data[category] = outflow_val
            
            if outflow_data:
                fig_outflow_pie = go.Figure(data=[go.Pie(
                    labels=list(outflow_data.keys()),
                    values=list(outflow_data.values()),
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Pastel2),
                    hovertemplate="%{label}<br>$%{value:.0f}M<br>%{percent}"
                )])
                
                fig_outflow_pie.update_layout(
                    title=f"Outflow Composition - {selected_date.strftime('%b %Y')}",
                    height=350
                )
                st.plotly_chart(fig_outflow_pie, use_container_width=True)
        
        with col3:
            # Net flow composition pie chart
            net_data = {}
            for category in categories:
                if category in selected_data:
                    net_val = selected_data[category]
                    if net_val != 0:
                        net_data[category] = abs(net_val)
            
            if net_data:
                fig_net_pie = go.Figure(data=[go.Pie(
                    labels=list(net_data.keys()),
                    values=list(net_data.values()),
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    hovertemplate="%{label}<br>$%{value:.0f}M<br>%{percent}"
                )])
                
                fig_net_pie.update_layout(
                    title=f"Absolute Flow Composition - {selected_date.strftime('%b %Y')}",
                    height=350
                )
                st.plotly_chart(fig_net_pie, use_container_width=True)
    
    # Monthly changes analysis
    if show_monthly_changes:
        st.markdown("### Monthly Percentage Changes")
        
        changes_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Monthly Percentage Changes', 'Cumulative Effect of Changes'],
            vertical_spacing=0.15
        )
        
        # Monthly percentage changes
        for i, category in enumerate(categories):
            if f'{category}_Monthly_Change' in df_filtered.columns:
                changes_fig.add_trace(
                    go.Scatter(
                        x=df_filtered['Date'],
                        y=df_filtered[f'{category}_Monthly_Change'],
                        name=category,
                        mode='lines',
                        line=dict(width=2),
                        hovertemplate='%{x|%b %Y}<br>' +
                                    f'{category}: %{{y:.1f}}%'
                    ),
                    row=1, col=1
                )
        
        # Cumulative effect
        for category in categories:
            if category in df_filtered.columns:
                cumulative_effect = (1 + df_filtered[category].pct_change()).cumprod() * 100
                changes_fig.add_trace(
                    go.Scatter(
                        x=df_filtered['Date'],
                        y=cumulative_effect,
                        name=f"{category} Cumulative",
                        mode='lines',
                        line=dict(width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        changes_fig.update_layout(
            height=700,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        changes_fig.update_yaxes(title_text="Monthly % Change", row=1, col=1)
        changes_fig.update_yaxes(title_text="Cumulative Index (Base=100)", row=2, col=1)
        
        st.plotly_chart(changes_fig, use_container_width=True)

with tab3:
    # Inflow/Outflow Analysis
    st.markdown('<div class="section-header">📊 Inflow vs Outflow Dynamics</div>', unsafe_allow_html=True)
    
    # Separate inflow and outflow analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monthly Inflows")
        
        # Prepare inflow data
        inflow_df = pd.DataFrame({'Date': df_filtered['Date']})
        for category in categories:
            if f'{category}_Inflow' in df_filtered.columns:
                inflow_df[category] = df_filtered[f'{category}_Inflow']
        
        fig_inflows = go.Figure()
        
        for category in categories:
            if category in inflow_df.columns:
                fig_inflows.add_trace(go.Bar(
                    name=category,
                    x=inflow_df['Date'],
                    y=inflow_df[category],
                    opacity=0.7,
                    hovertemplate='%{x|%b %Y}<br>' +
                                f'{category}: $%{{y:.0f}}M<br>' +
                                '<extra></extra>'
                ))
        
        fig_inflows.update_layout(
            barmode='stack',
            title="Monthly Inflows by Category",
            xaxis_title="Date",
            yaxis_title="Inflows (Millions USD)",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_inflows, use_container_width=True)
        
        # Inflow statistics
        st.markdown("**Inflow Statistics**")
        inflow_stats = pd.DataFrame({
            'Category': categories,
            'Total Inflow': [f"${df_filtered[f'{cat}_Inflow'].sum():,.0f}M" 
                           if f'{cat}_Inflow' in df_filtered.columns else "-" 
                           for cat in categories],
            'Avg Monthly': [f"${df_filtered[f'{cat}_Inflow'].mean():,.0f}M" 
                          if f'{cat}_Inflow' in df_filtered.columns else "-" 
                          for cat in categories],
            'Max Monthly': [f"${df_filtered[f'{cat}_Inflow'].max():,.0f}M" 
                          if f'{cat}_Inflow' in df_filtered.columns else "-" 
                          for cat in categories]
        })
        st.dataframe(inflow_stats, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Monthly Outflows")
        
        # Prepare outflow data
        outflow_df = pd.DataFrame({'Date': df_filtered['Date']})
        for category in categories:
            if f'{category}_Outflow' in df_filtered.columns:
                outflow_df[category] = df_filtered[f'{category}_Outflow']
        
        fig_outflows = go.Figure()
        
        for category in categories:
            if category in outflow_df.columns:
                fig_outflows.add_trace(go.Bar(
                    name=category,
                    x=outflow_df['Date'],
                    y=outflow_df[category],
                    opacity=0.7,
                    hovertemplate='%{x|%b %Y}<br>' +
                                f'{category}: $%{{y:.0f}}M<br>' +
                                '<extra></extra>'
                ))
        
        fig_outflows.update_layout(
            barmode='stack',
            title="Monthly Outflows by Category",
            xaxis_title="Date",
            yaxis_title="Outflows (Millions USD)",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_outflows, use_container_width=True)
        
        # Outflow statistics
        st.markdown("**Outflow Statistics**")
        outflow_stats = pd.DataFrame({
            'Category': categories,
            'Total Outflow': [f"${df_filtered[f'{cat}_Outflow'].sum():,.0f}M" 
                            if f'{cat}_Outflow' in df_filtered.columns else "-" 
                            for cat in categories],
            'Avg Monthly': [f"${df_filtered[f'{cat}_Outflow'].mean():,.0f}M" 
                          if f'{cat}_Outflow' in df_filtered.columns else "-" 
                          for cat in categories],
            'Max Monthly': [f"${df_filtered[f'{cat}_Outflow'].max():,.0f}M" 
                          if f'{cat}_Outflow' in df_filtered.columns else "-" 
                          for cat in categories]
        })
        st.dataframe(outflow_stats, use_container_width=True, hide_index=True)
    
    # Net flow analysis
    st.markdown("---")
    st.markdown("#### Net Flow Analysis")
    
    net_analysis_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Net Flow Over Time', 'Inflow-Outflow Balance',
                       'Cumulative Net Flow', 'Flow Direction Indicator'],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Net flow over time
    for category in categories:
        if category in df_filtered.columns:
            net_analysis_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered[category],
                    name=category,
                    mode='lines',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    # Inflow-outflow balance
    if 'Total' in df_filtered.columns:
        balance = df_filtered['Total_Inflow'] - df_filtered['Total_Outflow']
        net_analysis_fig.add_trace(
            go.Bar(
                x=df_filtered['Date'],
                y=balance,
                name='Net Balance',
                marker_color=['#10B981' if x > 0 else '#EF4444' for x in balance]
            ),
            row=1, col=2
        )
    
    # Cumulative net flow
    for category in categories:
        if f'{category}_Cumulative' in df_filtered.columns:
            net_analysis_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered[f'{category}_Cumulative'],
                    name=f'{category} Cumulative',
                    mode='lines',
                    line=dict(width=2, dash='dash')
                ),
                row=2, col=1
            )
    
    # Flow direction indicator
    direction_data = []
    for category in categories:
        if category in df_filtered.columns:
            inflow_months = (df_filtered[category] > 0).sum()
            outflow_months = (df_filtered[category] < 0).sum()
            direction_data.append(go.Bar(
                name=category,
                x=['Inflow', 'Outflow'],
                y=[inflow_months, outflow_months],
                text=[inflow_months, outflow_months],
                textposition='auto'
            ))
    
    for trace in direction_data:
        net_analysis_fig.add_trace(trace, row=2, col=2)
    
    net_analysis_fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    net_analysis_fig.update_yaxes(title_text="Net Flow (M USD)", row=1, col=1)
    net_analysis_fig.update_yaxes(title_text="Balance (M USD)", row=1, col=2)
    net_analysis_fig.update_yaxes(title_text="Cumulative (M USD)", row=2, col=1)
    net_analysis_fig.update_yaxes(title_text="Number of Months", row=2, col=2)
    
    st.plotly_chart(net_analysis_fig, use_container_width=True)

with tab4:
    # Composition Analysis
    st.markdown('<div class="section-header">🥧 Fund Composition Analysis</div>', unsafe_allow_html=True)
    
    # Time period selection for composition
    time_period = st.selectbox(
        "Select time period for composition analysis",
        ["Overall", "Yearly", "Quarterly", "Monthly"],
        index=0
    )
    
    if time_period == "Overall":
        # Overall composition
        st.markdown("### Overall Composition (Entire Period)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Total inflows composition
            total_inflows = {}
            for category in categories:
                if f'{category}_Inflow' in df_filtered.columns:
                    total_inflows[category] = df_filtered[f'{category}_Inflow'].sum()
            
            if total_inflows:
                fig_total_in = go.Figure(data=[go.Pie(
                    labels=list(total_inflows.keys()),
                    values=list(total_inflows.values()),
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    textinfo='label+percent',
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}"
                )])
                
                fig_total_in.update_layout(
                    title="Total Inflows Composition",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_total_in, use_container_width=True)
        
        with col2:
            # Total outflows composition
            total_outflows = {}
            for category in categories:
                if f'{category}_Outflow' in df_filtered.columns:
                    total_outflows[category] = df_filtered[f'{category}_Outflow'].sum()
            
            if total_outflows:
                fig_total_out = go.Figure(data=[go.Pie(
                    labels=list(total_outflows.keys()),
                    values=list(total_outflows.values()),
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    textinfo='label+percent',
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}"
                )])
                
                fig_total_out.update_layout(
                    title="Total Outflows Composition",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_total_out, use_container_width=True)
        
        with col3:
            # Net flows composition
            net_flows = {}
            for category in categories:
                if category in df_filtered.columns:
                    net_flows[category] = df_filtered[category].sum()
            
            if net_flows:
                fig_net = go.Figure(data=[go.Pie(
                    labels=list(net_flows.keys()),
                    values=[abs(v) for v in net_flows.values()],
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Bold),
                    textinfo='label+percent',
                    hovertemplate="%{label}<br>$%{value:,.0f}M<br>%{percent}"
                )])
                
                fig_net.update_layout(
                    title="Absolute Net Flows Composition",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_net, use_container_width=True)
    
    elif time_period == "Yearly":
        # Yearly composition
        st.markdown("### Yearly Composition Analysis")
        
        years = sorted(df_filtered['Year'].unique())
        selected_year = st.selectbox("Select Year", years, index=len(years)-1)
        
        year_data = df_filtered[df_filtered['Year'] == selected_year]
        
        if not year_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Yearly inflows
                yearly_inflows = {}
                for category in categories:
                    if f'{category}_Inflow' in year_data.columns:
                        yearly_inflows[category] = year_data[f'{category}_Inflow'].sum()
                
                if yearly_inflows:
                    fig_year_in = go.Figure(data=[go.Pie(
                        labels=list(yearly_inflows.keys()),
                        values=list(yearly_inflows.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set3)
                    )])
                    
                    fig_year_in.update_layout(
                        title=f"Inflows Composition - {selected_year}",
                        height=400
                    )
                    st.plotly_chart(fig_year_in, use_container_width=True)
            
            with col2:
                # Yearly outflows
                yearly_outflows = {}
                for category in categories:
                    if f'{category}_Outflow' in year_data.columns:
                        yearly_outflows[category] = year_data[f'{category}_Outflow'].sum()
                
                if yearly_outflows:
                    fig_year_out = go.Figure(data=[go.Pie(
                        labels=list(yearly_outflows.keys()),
                        values=list(yearly_outflows.values()),
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Pastel)
                    )])
                    
                    fig_year_out.update_layout(
                        title=f"Outflows Composition - {selected_year}",
                        height=400
                    )
                    st.plotly_chart(fig_year_out, use_container_width=True)
    
    # Composition trends over time
    st.markdown("### Composition Trends Over Time")
    
    # Calculate rolling composition
    window_size = st.slider("Rolling Window Size (months)", 3, 24, 12)
    
    composition_trends = pd.DataFrame({'Date': df_filtered['Date']})
    
    for category in categories:
        if category in df_filtered.columns:
            rolling_sum = df_filtered[category].rolling(window=window_size).sum()
            total_rolling = df_filtered[categories].rolling(window=window_size).sum().sum(axis=1)
            composition_trends[f'{category}_%'] = 100 * rolling_sum / total_rolling
    
    fig_composition_trends = go.Figure()
    
    for category in categories:
        if f'{category}_%' in composition_trends.columns:
            fig_composition_trends.add_trace(go.Scatter(
                x=composition_trends['Date'],
                y=composition_trends[f'{category}_%'],
                name=category,
                mode='lines',
                stackgroup='one',
                hovertemplate='%{x|%b %Y}<br>' +
                            f'{category}: %{{y:.1f}}%<br>' +
                            '<extra></extra>'
            ))
    
    fig_composition_trends.update_layout(
        title=f"Rolling {window_size}-Month Composition Trends",
        xaxis_title="Date",
        yaxis_title="Percentage of Total Flow",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_composition_trends, use_container_width=True)

with tab5:
    # Performance Metrics
    st.markdown('<div class="section-header">📉 Performance Metrics & Analytics</div>', unsafe_allow_html=True)
    
    # Key performance indicators
    st.markdown("### Key Performance Indicators")
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        if 'Total' in df_filtered.columns:
            sharpe_ratio = df_filtered['Total'].mean() / df_filtered['Total'].std() * np.sqrt(12)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    
    with kpi_cols[1]:
        if 'Total' in df_filtered.columns:
            sortino_ratio = df_filtered['Total'].mean() / df_filtered[df_filtered['Total'] < 0]['Total'].std() * np.sqrt(12)
            st.metric("Sortino Ratio", f"{sortino_ratio:.3f}")
    
    with kpi_cols[2]:
        if 'Total' in df_filtered.columns:
            max_drawdown = (df_filtered['Total_Cumulative'].cummax() - df_filtered['Total_Cumulative']).max()
            st.metric("Max Drawdown", f"${max_drawdown:,.0f}M")
    
    with kpi_cols[3]:
        if 'Total' in df_filtered.columns:
            win_rate = (df_filtered['Total'] > 0).sum() / len(df_filtered) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Correlation analysis
    st.markdown("### Correlation Analysis")
    
    # Calculate correlations
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
            title="Correlation Matrix Between Fund Categories",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Volatility analysis
    st.markdown("### Volatility Analysis")
    
    vol_cols = st.columns(2)
    
    with vol_cols[0]:
        # Rolling volatility
        if 'Total' in df_filtered.columns:
            rolling_vol = df_filtered['Total'].rolling(window=12).std()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df_filtered['Date'],
                y=rolling_vol,
                name='12-Month Rolling Volatility',
                line=dict(color='red', width=2)
            ))
            
            fig_vol.update_layout(
                title="12-Month Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (Std Dev)",
                height=300
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with vol_cols[1]:
        # Volatility by category
        vol_by_category = {}
        for category in categories:
            if category in df_filtered.columns:
                vol_by_category[category] = df_filtered[category].std()
        
        if vol_by_category:
            fig_vol_cat = go.Figure(data=[go.Bar(
                x=list(vol_by_category.keys()),
                y=list(vol_by_category.values()),
                marker_color=px.colors.qualitative.Set3
            )])
            
            fig_vol_cat.update_layout(
                title="Volatility by Category",
                xaxis_title="Category",
                yaxis_title="Standard Deviation",
                height=300
            )
            st.plotly_chart(fig_vol_cat, use_container_width=True)
    
    # Distribution analysis
    st.markdown("### Distribution Analysis")
    
    dist_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Distribution by Category', 'QQ-Plot (Normality Test)']
    )
    
    # Histograms
    for category in categories:
        if category in df_filtered.columns:
            dist_fig.add_trace(
                go.Histogram(
                    x=df_filtered[category],
                    name=category,
                    opacity=0.6,
                    nbinsx=30
                ),
                row=1, col=1
            )
    
    # QQ-Plot for normality
    if 'Total' in df_filtered.columns:
        # Calculate theoretical quantiles
        theoretical_quantiles = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(df_filtered))))
        sample_quantiles = np.sort(df_filtered['Total'].values)
        
        dist_fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='QQ-Plot',
                marker=dict(size=6, color='blue')
            ),
            row=1, col=2
        )
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        dist_fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
    
    dist_fig.update_layout(
        height=400,
        showlegend=True,
        barmode='overlay'
    )
    
    st.plotly_chart(dist_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Investment Company Institute (ICI) Mutual Fund Flows Dashboard</strong></p>
    <p>Data Source: ICI Statistics | All figures in millions of USD | Negative values indicate net outflows</p>
    <p>Dashboard Version 2.0 | Last Updated: Sample Data Generated</p>
</div>
""", unsafe_allow_html=True)
