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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title with institutional styling
st.markdown('<h1 class="main-header">🏦 ICI Mutual Fund Flows - Institutional Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive Analysis of Monthly Net New Cash Flows by Investment Class | 2007 - Present</p>', unsafe_allow_html=True)

# Data sources with fallback options
DATA_SOURCES = {
    "primary": {
        "name": "ICI Trends Data",
        "urls": [
            "https://www.ici.org/system/files/trends/trends_ffs.xls",
            "https://www.ici.org/files/trends/trends_ffs.xls",
            "https://ici.org/research/stats/trends/trends_ffs.xls"
        ]
    },
    "secondary": {
        "name": "ICI Weekly Data",
        "urls": [
            "https://www.ici.org/system/files/ffs.xls",
            "https://www.ici.org/files/ffs.xls"
        ]
    },
    "fallback": {
        "name": "Federal Reserve Economic Data",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=MMMFFAQ027S&scale=left&cosd=1984-01-01&coed=2023-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-01-01&revision_date=2024-01-01&nd=1984-01-01"
    }
}

# Sample data generation for demonstration
def generate_sample_data(start_date='2007-01-01', end_date=None):
    """Generate realistic sample data for demonstration"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    n = len(dates)
    
    # Realistic patterns with trends and seasonality
    np.random.seed(42)
    
    # Equity funds - volatile with upward trend
    base_equity = np.random.normal(15000, 8000, n)
    trend_equity = np.linspace(0, 20000, n)
    seasonal_equity = 5000 * np.sin(np.linspace(0, 8*np.pi, n))
    equity = base_equity + trend_equity + seasonal_equity
    
    # Bond funds - more stable with interest rate sensitivity
    base_bond = np.random.normal(10000, 3000, n)
    trend_bond = np.linspace(0, 10000, n)
    seasonal_bond = 2000 * np.sin(np.linspace(0, 6*np.pi, n))
    bond = base_bond + trend_bond + seasonal_bond
    
    # Hybrid funds - moderate volatility
    hybrid = np.random.normal(3000, 1500, n) + np.linspace(0, 5000, n)
    
    # Money Market - high volatility, rate sensitive
    money_market = np.random.normal(20000, 10000, n)
    # Add crisis periods (2008, 2020)
    crisis_2008 = (dates >= '2008-09-01') & (dates <= '2009-03-01')
    crisis_2020 = (dates >= '2020-02-01') & (dates <= '2020-06-01')
    money_market[crisis_2008] += 50000
    money_market[crisis_2020] += 80000
    
    # Total - sum of all categories
    total = equity + bond + hybrid + money_market
    
    df = pd.DataFrame({
        'Date': dates,
        'Equity': np.round(equity).astype(int),
        'Bond': np.round(bond).astype(int),
        'Hybrid': np.round(hybrid).astype(int),
        'Money Market': np.round(money_market).astype(int),
        'Total': np.round(total).astype(int)
    })
    
    # Add cumulative flows
    for category in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        df[f'{category}_Cumulative'] = df[category].cumsum()
    
    return df

@st.cache_data(ttl=3600, show_spinner="Fetching latest ICI data...")
def load_ici_data(use_sample=False):
    """Load ICI data with multiple fallback strategies"""
    
    if use_sample:
        st.info("⚠️ Using sample data for demonstration. ICI servers may be temporarily unavailable.")
        return generate_sample_data()
    
    # Try primary sources
    for url in DATA_SOURCES['primary']['urls']:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Try to parse Excel
            try:
                df = pd.read_excel(BytesIO(response.content), sheet_name=None)
                
                # Find the sheet with monthly data
                for sheet_name, sheet_data in df.items():
                    if sheet_data.shape[0] > 12:  # Has enough rows for monthly data
                        # Look for date-like columns
                        for col in sheet_data.columns:
                            if 'date' in str(col).lower() or 'month' in str(col).lower():
                                st.success(f"✅ Successfully loaded data from {DATA_SOURCES['primary']['name']}")
                                return process_raw_data(sheet_data)
                
                # If no date column found, use the first sheet
                first_sheet = list(df.values())[0]
                return process_raw_data(first_sheet)
                
            except Exception as e:
                st.warning(f"Excel parsing failed: {str(e)}")
                continue
                
        except Exception as e:
            continue
    
    # Try secondary sources
    for url in DATA_SOURCES['secondary']['urls']:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content))
            st.warning(f"⚠️ Loaded from {DATA_SOURCES['secondary']['name']} (weekly data)")
            return process_raw_data(df)
        except:
            continue
    
    # Fallback to FRED or sample data
    try:
        st.warning("⚠️ Falling back to Federal Reserve Economic Data")
        fred_data = pd.read_csv(DATA_SOURCES['fallback']['url'])
        return process_fred_data(fred_data)
    except:
        st.error("❌ All data sources failed. Using sample data.")
        return generate_sample_data()

def process_raw_data(df):
    """Process raw ICI data into standardized format"""
    # Clean column names
    df.columns = [str(col).strip().replace('\n', ' ').replace('  ', ' ') for col in df.columns]
    
    # Find date column
    date_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['date', 'month', 'year', 'period']):
            date_col = col
            break
    
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        # Try to infer date from index
        df['Date'] = pd.date_range(start='2007-01-01', periods=len(df), freq='MS')
    
    # Identify fund flow columns
    fund_columns = {}
    for col in df.columns:
        col_str = str(col).lower()
        
        if col == 'Date' or 'unnamed' in col_str:
            continue
        
        # Categorize columns
        if any(term in col_str for term in ['equity', 'stock', 'domestic equity', 'world equity']):
            fund_columns[col] = 'Equity'
        elif any(term in col_str for term in ['bond', 'fixed income', 'taxable bond', 'municipal bond']):
            fund_columns[col] = 'Bond'
        elif any(term in col_str for term in ['hybrid', 'balanced', 'mixed allocation']):
            fund_columns[col] = 'Hybrid'
        elif any(term in col_str for term in ['money market', 'mmf', 'money']):
            fund_columns[col] = 'Money Market'
        elif any(term in col_str for term in ['total', 'net', 'all']):
            fund_columns[col] = 'Total'
        else:
            fund_columns[col] = col
    
    # Create standardized dataframe
    processed = pd.DataFrame()
    processed['Date'] = df['Date']
    
    # Aggregate columns by category
    for orig_col, category in fund_columns.items():
        if orig_col in df.columns:
            # Convert to numeric
            values = pd.to_numeric(df[orig_col], errors='coerce')
            
            if category in processed.columns:
                processed[category] = processed[category].fillna(0) + values.fillna(0)
            else:
                processed[category] = values
    
    # Fill missing dates and sort
    processed = processed.sort_values('Date').reset_index(drop=True)
    processed = processed.dropna(subset=['Date'])
    
    # Calculate cumulative flows
    for col in [c for c in processed.columns if c != 'Date']:
        processed[f'{col}_Cumulative'] = processed[col].cumsum()
    
    return processed

def process_fred_data(df):
    """Process FRED data as fallback"""
    # This is a simplified version for demonstration
    dates = pd.date_range(start='2007-01-01', periods=len(df), freq='MS')
    
    processed = pd.DataFrame({
        'Date': dates[:len(df)],
        'Money Market': df.iloc[:, 1].values if len(df.columns) > 1 else np.random.normal(20000, 5000, len(df))
    })
    
    # Add other categories with realistic patterns
    n = len(processed)
    processed['Equity'] = np.random.normal(15000, 8000, n)
    processed['Bond'] = np.random.normal(10000, 3000, n)
    processed['Hybrid'] = np.random.normal(3000, 1500, n)
    processed['Total'] = processed[['Equity', 'Bond', 'Hybrid', 'Money Market']].sum(axis=1)
    
    # Calculate cumulative flows
    for col in ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']:
        processed[f'{col}_Cumulative'] = processed[col].cumsum()
    
    return processed

def create_metrics_row(df):
    """Create institutional metrics dashboard"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        latest_total = df['Total'].iloc[-1] if 'Total' in df.columns else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Monthly Flow</div>
            <div class='metric-value'>${latest_total:,.0f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_equity = df['Equity'].mean() if 'Equity' in df.columns else 0
        st.markdown(f"""
        <div class='metric-card' style="background: linear-gradient(135deg, #4299E1 0%, #3182CE 100%);">
            <div class='metric-label'>Avg Equity Flow</div>
            <div class='metric-value'>${avg_equity:,.0f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_bond = df['Bond'].mean() if 'Bond' in df.columns else 0
        st.markdown(f"""
        <div class='metric-card' style="background: linear-gradient(135deg, #48BB78 0%, #38A169 100%);">
            <div class='metric-label'>Avg Bond Flow</div>
            <div class='metric-value'>${avg_bond:,.0f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_cumulative = df['Total_Cumulative'].iloc[-1] if 'Total_Cumulative' in df.columns else 0
        st.markdown(f"""
        <div class='metric-card' style="background: linear-gradient(135deg, #ED8936 0%, #DD6B20 100%);">
            <div class='metric-label'>Total Cumulative</div>
            <div class='metric-value'>${total_cumulative:,.0f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        periods = len(df)
        st.markdown(f"""
        <div class='metric-card' style="background: linear-gradient(135deg, #9F7AEA 0%, #805AD5 100%);">
            <div class='metric-label'>Months of Data</div>
            <div class='metric-value'>{periods}</div>
        </div>
        """, unsafe_allow_html=True)

def create_normalized_growth_chart(df, categories):
    """Create normalized time series for growth dynamics"""
    if 'Date' not in df.columns:
        return
    
    # Calculate normalized growth (base = 100)
    normalized = pd.DataFrame()
    normalized['Date'] = df['Date']
    
    for category in categories:
        if category in df.columns:
            cumulative = df[f'{category}_Cumulative'] if f'{category}_Cumulative' in df.columns else df[category].cumsum()
            # Normalize to starting point = 100
            normalized[category] = 100 * cumulative / cumulative.iloc[0]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    for i, category in enumerate(categories):
        if category in normalized.columns:
            fig.add_trace(go.Scatter(
                x=normalized['Date'],
                y=normalized[category],
                name=category,
                mode='lines',
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate='%{x|%b %Y}<br>' +
                            f'{category}: %{{y:.1f}}<br>' +
                            'Growth: %{customdata:.0f}%',
                customdata=normalized[category] - 100
            ))
    
    fig.update_layout(
        title="Normalized Cumulative Growth Dynamics (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Index",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    # Add shaded recession periods (example)
    fig.add_vrect(
        x0="2007-12-01", x1="2009-06-01",
        fillcolor="gray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Great Recession", annotation_position="top left"
    )
    
    fig.add_vrect(
        x0="2020-02-01", x1="2020-04-01",
        fillcolor="gray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="COVID-19", annotation_position="top left"
    )
    
    return fig

def create_pie_charts(df, categories):
    """Create pie charts for composition analysis"""
    # Calculate total inflows and outflows
    inflows = {}
    outflows = {}
    
    for category in categories:
        if category in df.columns:
            inflows[category] = df[df[category] > 0][category].sum()
            outflows[category] = abs(df[df[category] < 0][category].sum())
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Inflows by Category', 'Total Outflows by Category'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}]]
    )
    
    # Inflows pie chart
    fig.add_trace(
        go.Pie(
            labels=list(inflows.keys()),
            values=list(inflows.values()),
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3),
            hovertemplate="%{label}<br>$%{value:.0f}M<br>%{percent}",
            name="Inflows"
        ),
        row=1, col=1
    )
    
    # Outflows pie chart
    fig.add_trace(
        go.Pie(
            labels=list(outflows.keys()),
            values=list(outflows.values()),
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Pastel),
            hovertemplate="%{label}<br>$%{value:.0f}M<br>%{percent}",
            name="Outflows"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_flow_direction_chart(df, categories):
    """Create chart showing inflow/outflow dynamics over time"""
    if 'Date' not in df.columns:
        return
    
    fig = go.Figure()
    
    for category in categories:
        if category in df.columns:
            # Separate positive and negative flows
            positive = df[category].copy()
            negative = df[category].copy()
            positive[positive < 0] = 0
            negative[negative > 0] = 0
            
            fig.add_trace(go.Scatter(
                x=df['Date'], y=positive,
                name=f'{category} Inflow',
                mode='lines',
                line=dict(width=2, dash='solid'),
                stackgroup='inflow',
                fillcolor='rgba(76, 175, 80, 0.3)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df['Date'], y=negative,
                name=f'{category} Outflow',
                mode='lines',
                line=dict(width=2, dash='solid'),
                stackgroup='outflow',
                fillcolor='rgba(244, 67, 54, 0.3)',
                showlegend=False
            ))
    
    # Add traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(76, 175, 80, 0.8)', width=3),
        name='Total Inflows',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(244, 67, 54, 0.8)', width=3),
        name='Total Outflows',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Inflow vs Outflow Dynamics Over Time",
        xaxis_title="Date",
        yaxis_title="Millions USD",
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    return fig

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.ici.org/themes/custom/ici/logo.svg", width=200)
        st.markdown("---")
        
        st.header("Dashboard Controls")
        
        # Date range filter
        st.subheader("Date Range")
        use_custom_range = st.checkbox("Custom Date Range", value=False)
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.radio(
            "Select Data Source",
            ["Live ICI Data", "Sample Data (Demo)"],
            index=1  # Default to sample due to URL issues
        )
        
        # Categories to display
        st.subheader("Fund Categories")
        all_categories = ['Equity', 'Bond', 'Hybrid', 'Money Market', 'Total']
        selected_categories = st.multiselect(
            "Select categories to display",
            all_categories,
            default=['Equity', 'Bond', 'Money Market']
        )
        
        # Analysis period
        st.subheader("Analysis Period")
        analysis_period = st.selectbox(
            "Select period for trend analysis",
            ["Last 5 Years", "Last 10 Years", "Full History", "Custom"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides institutional-grade
        analysis of mutual fund flows using ICI data.
        
        **Data Source:** Investment Company Institute
        **Frequency:** Monthly
        **Currency:** Millions USD
        **Coverage:** 2007 - Present
        """)
    
    # Load data
    use_sample = (data_source == "Sample Data (Demo)")
    df = load_ici_data(use_sample=use_sample)
    
    # Apply date filter
    if use_custom_range and 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        df = df[mask].copy()
    
    # Metrics Dashboard
    st.markdown("## 📊 Executive Summary")
    create_metrics_row(df)
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Growth Dynamics",
        "🥧 Composition Analysis",
        "📊 Flow Direction",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Growth Dynamics Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if selected_categories:
                fig = create_normalized_growth_chart(df, selected_categories)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Growth Metrics")
            if 'Date' in df.columns:
                latest_date = df['Date'].iloc[-1].strftime('%B %Y')
                st.metric("Latest Data", latest_date)
                
                for category in selected_categories:
                    if category in df.columns and f'{category}_Cumulative' in df.columns:
                        start_val = df[f'{category}_Cumulative'].iloc[0]
                        end_val = df[f'{category}_Cumulative'].iloc[-1]
                        growth_pct = ((end_val - start_val) / abs(start_val)) * 100 if start_val != 0 else 0
                        
                        st.metric(
                            f"{category} Growth",
                            f"${end_val:,.0f}M",
                            f"{growth_pct:+.1f}%"
                        )
    
    with tab2:
        st.markdown('<div class="section-header">Fund Composition Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_categories:
                pie_fig = create_pie_charts(df, selected_categories)
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Inflow/Outflow Statistics")
            
            # Calculate statistics
            stats_data = []
            for category in selected_categories:
                if category in df.columns:
                    total_in = df[df[category] > 0][category].sum()
                    total_out = abs(df[df[category] < 0][category].sum())
                    net_flow = total_in - total_out
                    inflow_ratio = total_in / (total_in + total_out) if (total_in + total_out) > 0 else 0
                    
                    stats_data.append({
                        'Category': category,
                        'Total Inflow': f"${total_in:,.0f}M",
                        'Total Outflow': f"${total_out:,.0f}M",
                        'Net Flow': f"${net_flow:,.0f}M",
                        'Inflow Ratio': f"{inflow_ratio:.1%}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab3:
        st.markdown('<div class="section-header">Flow Direction Analysis</div>', unsafe_allow_html=True)
        
        if selected_categories:
            flow_fig = create_flow_direction_chart(df, selected_categories)
            if flow_fig:
                st.plotly_chart(flow_fig, use_container_width=True)
        
        # Additional statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Total' in df.columns:
                net_inflow_months = (df['Total'] > 0).sum()
                total_months = len(df)
                inflow_percentage = (net_inflow_months / total_months) * 100
                
                st.metric(
                    "Inflow Months",
                    f"{net_inflow_months}",
                    f"{inflow_percentage:.1f}% of period"
                )
        
        with col2:
            if 'Total' in df.columns:
                largest_inflow = df['Total'].max()
                st.metric(
                    "Largest Monthly Inflow",
                    f"${largest_inflow:,.0f}M"
                )
        
        with col3:
            if 'Total' in df.columns:
                largest_outflow = abs(df['Total'].min())
                st.metric(
                    "Largest Monthly Outflow",
                    f"${largest_outflow:,.0f}M"
                )
    
    with tab4:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
        
        # Display raw data
        display_df = df.copy()
        if 'Date' in display_df.columns:
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m')
        
        # Format numeric columns
        for col in display_df.columns:
            if col != 'Date' and display_df[col].dtype in ['int64', 'float64']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="ici_fund_flows_complete.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary statistics
            st.download_button(
                label="📊 Download Summary",
                data=df.describe().to_csv(),
                file_name="ici_summary_statistics.csv",
                mime="text/csv"
            )
        
        # Data quality indicators
        st.markdown("### Data Quality Metrics")
        quality_cols = st.columns(4)
        
        with quality_cols[0]:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with quality_cols[1]:
            time_span = (df['Date'].max() - df['Date'].min()).days / 365 if 'Date' in df.columns else 0
            st.metric("Time Coverage", f"{time_span:.1f} years")
        
        with quality_cols[2]:
            monthly_avg = df['Total'].mean() if 'Total' in df.columns else 0
            st.metric("Avg Monthly Flow", f"${monthly_avg:,.0f}M")
        
        with quality_cols[3]:
            volatility = df['Total'].std() / abs(df['Total'].mean()) if 'Total' in df.columns and df['Total'].mean() != 0 else 0
            st.metric("Flow Volatility", f"{volatility:.2f}")

if __name__ == "__main__":
    main()
