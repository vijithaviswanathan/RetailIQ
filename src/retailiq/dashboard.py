import os
import sys
import textwrap
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sqlalchemy import create_engine, text


# ============================================================
# PROJECT IMPORT CONFIGURATION
# ============================================================

# Add the project's src directory to Python's import path.
PROJECT_SRC = str(Path(__file__).resolve().parents[1])

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetailIQ | Decision Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

from retailiq.config import DATABASE_URL


# ============================================================
# API CONFIGURATION
# ============================================================

ASSISTANT_URL = os.getenv(
    "RETAILIQ_ASSISTANT_URL",
    "http://127.0.0.1:8000/assistant/ask",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

COLORS = {
    "blue": "#8DB7E8",
    "blue_dark": "#477DBD",
    "sage": "#A9CEB3",
    "sage_dark": "#548B69",
    "lavender": "#C8B8E8",
    "lavender_dark": "#8063B4",
    "peach": "#F2C5A8",
    "peach_dark": "#B87548",
    "teal": "#9CCFCB",
    "pink": "#E8B7C8",
    "navy": "#26364A",
    "text": "#344256",
    "muted": "#526176",
    "border": "#D8E1EB",
    "background": "#F7F9FC",
}

CHART_COLORS = [
    COLORS["blue_dark"],
    COLORS["sage_dark"],
    COLORS["lavender_dark"],
    COLORS["peach_dark"],
    "#438F91",
    "#B56D89",
]


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    :root {
        color-scheme: light;
    }

    html,
    body,
    [class*="css"],
    [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif !important;
        color: #26364A !important;
    }

    .stApp {
        background: #F7F9FC !important;
        color: #26364A !important;
    }

    [data-testid="stHeader"] {
        background: #F7F9FC !important;
    }

    [data-testid="stMain"] {
        color: #26364A !important;
    }

    [data-testid="stMainBlockContainer"] {
        color: #26364A !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #D8E1EB !important;
    }

    [data-testid="stSidebar"] * {
        color: #26364A !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #344256 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #26364A !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 7px 3px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #26364A !important;
    }

    /* Headings */

    h1, h2, h3, h4, h5, h6 {
        color: #172B4D !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }

    h1 {
        font-size: 2rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    /* Body text */

    p,
    li,
    label,
    .stMarkdown,
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] {
        color: #344256 !important;
        opacity: 1 !important;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] p {
        color: #526176 !important;
        opacity: 1 !important;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #D8E1EB !important;
        border-radius: 15px !important;
        padding: 17px !important;
        box-shadow: 0 3px 12px rgba(35,55,80,0.045);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #172B4D !important;
        font-weight: 750 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        opacity: 1 !important;
    }

    /* Buttons */

    div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px !important;
        border: 1px solid #B8CDE2 !important;
        background: #FFFFFF !important;
        color: #244B75 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    div.stButton > button p,
    div.stButton > button span {
        color: #244B75 !important;
        opacity: 1 !important;
    }

    div.stButton > button:hover {
        border-color: #477DBD !important;
        background: #EAF2FC !important;
        color: #172B4D !important;
    }

    div.stButton > button:focus {
        border-color: #477DBD !important;
        color: #172B4D !important;
    }

    div.stButton > button[kind="primary"] {
        background: #477DBD !important;
        border-color: #477DBD !important;
        color: #FFFFFF !important;
    }

    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
    }

    /* Inputs */

    input,
    textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: #26364A !important;
        background: #FFFFFF !important;
        border-color: #B8C8D9 !important;
        opacity: 1 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #718096 !important;
        opacity: 1 !important;
    }

    [data-baseweb="select"] * {
        color: #26364A !important;
        opacity: 1 !important;
    }

    /* Select boxes and date controls */

    [data-testid="stDateInput"],
    [data-testid="stSelectbox"],
    [data-testid="stRadio"],
    [data-testid="stTextArea"] {
        color: #26364A !important;
    }

    /* Tables */

    [data-testid="stDataFrame"] {
        border: 1px solid #D8E1EB !important;
        border-radius: 12px !important;
    }

    [data-testid="stDataFrame"] * {
        color: #26364A !important;
    }

    /* Charts */

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF !important;
        border: 1px solid #D8E1EB !important;
        border-radius: 14px !important;
        padding: 7px !important;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #D8E1EB !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #26364A !important;
        font-weight: 650 !important;
    }

    /* Alerts */

    [data-testid="stAlert"] {
        background: #FFFFFF !important;
        border: 1px solid #D8E1EB !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(35, 55, 80, 0.04) !important;
        opacity: 1 !important;
    }

    [data-testid="stAlert"] > div {
        background: #FFFFFF !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #26364A !important;
        opacity: 1 !important;
    }

    /* Dividers */

    hr {
        border-color: #D8E1EB !important;
    }

    /* Hero */

    .hero {
        background: linear-gradient(
            115deg,
            #EAF2FC 0%,
            #F2EFFB 55%,
            #FDF1E8 100%
        );
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #D8E1EB;
        margin-bottom: 24px;
    }

    .hero-title {
        color: #172B4D !important;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #344256 !important;
        font-size: 15px;
        line-height: 1.7;
    }

    .section-label {
        color: #405875 !important;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    /* Module cards */

    .module-card {
        background: #FFFFFF;
        border: 1px solid #D8E1EB;
        border-radius: 16px;
        padding: 20px;
        min-height: 175px;
        box-shadow: 0 3px 12px rgba(35,55,80,0.045);
    }

    .module-icon {
        font-size: 25px;
        margin-bottom: 10px;
    }

    .module-title {
        color: #172B4D !important;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .module-description {
        color: #475569 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .footer-text {
        color: #526176 !important;
        text-align: center;
        font-size: 12px;
        padding: 10px;
    }

    /* Improve visibility of dashboard text */
    [data-testid="stApp"] { 
        color: #26364A;
    }

    [data-testid="stApp"] p,
    [data-testid="stApp"] span,
    [data-testid="stApp"] label,
    [data-testid="stApp"] small,
    [data-testid="stApp"] li,
    [data-testid="stApp"] [data-testid="stMarkdownContainer"] {
        color: #26364A;
        opacity: 1;
    }   

    /* Chart labels, legends, color-scale sidebars and annotations */
    [data-testid="stPlotlyChart"] .legend text,
    [data-testid="stPlotlyChart"] .xtick text,
    [data-testid="stPlotlyChart"] .ytick text,
    [data-testid="stPlotlyChart"] .colorbar text,
    [data-testid="stPlotlyChart"] .annotation-text {
        fill: #26364A !important;
    }

    /* Keep date-range values readable on the light dashboard */
    [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stDateInput"] [data-baseweb="input"] > div,
    [data-testid="stDateInput"] input,
    [data-baseweb="datepicker"] input {
        color: #26364A !important;
        -webkit-text-fill-color: #26364A !important;
        background-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stDateInput"] [data-baseweb="input"] {
        border-color: #B8C8D9 !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_data(query):
    with engine.connect() as connection:
        return pd.read_sql(
            text(query),
            connection,
        )


def clean_numeric_columns(data, columns):
    data = data.copy()

    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def wrap_labels(values, width=25):
    return [
        textwrap.fill(
            str(value),
            width=width,
            break_long_words=False,
        )
        for value in values
    ]


# ============================================================
# CHART HELPERS
# ============================================================

def chart_layout(
    fig,
    height=400,
    bottom_margin=85,
    left_margin=75,
    right_margin=55,
):
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        margin=dict(
            l=left_margin,
            r=right_margin,
            t=85,
            b=bottom_margin,
        ),
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#26364A",
        ),
        title=dict(
            font=dict(
                size=18,
                color="#172B4D",
            ),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            font=dict(
                size=12,
                color="#26364A",
            ),
            bgcolor="#FFFFFF",
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#AAB7C7",
            font=dict(
                size=13,
                color="#26364A",
            ),
        ),
        hovermode="closest",
        uniformtext=dict(
            minsize=10,
            mode="hide",
        ),
    )

    # Ensure continuous-color sidebars and small chart notes use
    # readable dark text against the white chart background.
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(size=12, color="#26364A"),
            title_font=dict(size=12, color="#26364A"),
        )
    )
    fig.update_annotations(
        font=dict(color="#26364A")
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E3EAF2",
        linecolor="#AAB7C7",
        showline=True,
        ticks="outside",
        tickfont=dict(
            size=12,
            color="#344256",
        ),
        title_font=dict(
            size=13,
            color="#26364A",
        ),
        automargin=True,
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E3EAF2",
        linecolor="#AAB7C7",
        showline=True,
        ticks="outside",
        tickfont=dict(
            size=12,
            color="#344256",
        ),
        title_font=dict(
            size=13,
            color="#26364A",
        ),
        automargin=True,
        zeroline=False,
    )

    return fig


def show_chart(
    fig,
    height=400,
    bottom_margin=85,
    left_margin=75,
    right_margin=55,
):
    fig = chart_layout(
        fig,
        height=height,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "retailiq_chart",
                "scale": 2,
            },
        },
    )


def page_header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()


def make_gauge(
    value,
    title,
    minimum=0,
    maximum=1,
    threshold=None,
):
    gauge_config = {
        "axis": {
            "range": [minimum, maximum],
            "tickcolor": "#344256",
            "tickfont": {
                "color": "#344256",
            },
        },
        "bar": {
            "color": COLORS["blue_dark"],
        },
        "bgcolor": "#FFFFFF",
        "borderwidth": 0,
        "steps": [
            {
                "range": [minimum, maximum],
                "color": "#EDF1F6",
            }
        ],
    }

    if threshold is not None:
        gauge_config["threshold"] = {
            "line": {
                "color": "#D97757",
                "width": 3,
            },
            "thickness": 0.8,
            "value": threshold,
        }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "font": {
                    "size": 32,
                    "color": "#172B4D",
                },
                "valueformat": ".4f",
            },
            title={
                "text": title,
                "font": {
                    "size": 16,
                    "color": "#344256",
                },
            },
            gauge=gauge_config,
        )
    )

    return chart_layout(
        fig,
        height=290,
        bottom_margin=35,
        left_margin=35,
        right_margin=35,
    )


def show_error(message, exc):
    st.error(f"{message}: {exc}")


# ============================================================
# NAVIGATION
# ============================================================

NAV_OPTIONS = [
    "Home",
    "Retail Analytics",
    "Experimentation",
    "Causal ML & Optimization",
    "Predictive ML",
    "Forecasting",
    "AI Assistant",
]

if "navigation_page" not in st.session_state:
    st.session_state["navigation_page"] = "Home"


def navigate_to(destination):
    """
    Streamlit executes this callback before rerunning the script.
    This safely updates the sidebar radio's session-state value.
    """
    st.session_state["navigation_page"] = destination


def load_suggested_question(question):
    """
    Update the text area from a callback before the next rerun.
    """
    st.session_state["assistant_input"] = question


with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 5px 22px 5px;">
            <div style="
                font-size: 26px;
                font-weight: 800;
                color: #172B4D;
            ">
                📊 RetailIQ
            </div>
            <div style="
                font-size: 12px;
                font-weight: 700;
                color: #526176;
                letter-spacing: 0.8px;
                margin-top: 5px;
            ">
                RETAIL DECISION INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="navigation_page",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div style="
            color: #344256;
            font-weight: 750;
            font-size: 13px;
        ">
            RETAILIQ PLATFORM
        </div>
        <div style="
            color: #526176;
            font-size: 12px;
            line-height: 1.8;
            margin-top: 7px;
        ">
            PostgreSQL · FastAPI<br>
            Machine Learning · AI
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="section-label">
                RETAIL DECISION INTELLIGENCE
            </div>
            <div class="hero-title">
                Welcome to RetailIQ
            </div>
            <div class="hero-subtitle">
                Explore retail performance, campaign experiments,
                predictive models and AI-powered business intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Executive Overview")
    st.caption(
        "Business metrics from your PostgreSQL analytics views."
    )

    try:
        kpis = get_data(
            "SELECT * FROM retail.executive_kpis"
        )

        if not kpis.empty:
            row = kpis.iloc[0]

            columns = st.columns(4)

            columns[0].metric(
                "Merchandise Revenue",
                f"£{row['total_merchandise_revenue']:,.0f}",
            )

            columns[1].metric(
                "Total Orders",
                f"{row['total_orders']:,.0f}",
            )

            columns[2].metric(
                "Identified Customers",
                f"{row['identified_customers']:,.0f}",
            )

            columns[3].metric(
                "Average Order Value",
                f"£{row['average_order_value']:,.2f}",
            )

            columns = st.columns(3)

            columns[0].metric(
                "Distinct Products",
                f"{row['distinct_products']:,.0f}",
            )

            columns[1].metric(
                "Units Sold",
                f"{row['total_units_sold']:,.0f}",
            )

            columns[2].metric(
                "Revenue Without Customer ID",
                f"£{row['revenue_without_customer_id']:,.0f}",
            )

        else:
            st.warning(
                "No executive KPI data is available."
            )

    except Exception as exc:
        show_error(
            "Could not load executive KPIs",
            exc,
        )

    st.divider()

    st.markdown("## Business Performance")

    try:
        sales = get_data(
            """
            SELECT
                month,
                total_orders,
                units_sold,
                merchandise_revenue,
                identified_customers
            FROM retail.monthly_sales
            ORDER BY month
            """
        )

        if not sales.empty:
            sales["month"] = pd.to_datetime(
                sales["month"]
            )

            sales = clean_numeric_columns(
                sales,
                [
                    "total_orders",
                    "units_sold",
                    "merchandise_revenue",
                    "identified_customers",
                ],
            )

            left, right = st.columns(2)

            with left:
                fig = px.area(
                    sales,
                    x="month",
                    y="merchandise_revenue",
                    title="Monthly Merchandise Revenue",
                    color_discrete_sequence=[
                        COLORS["blue_dark"]
                    ],
                )

                fig.update_traces(
                    line=dict(width=3),
                    fillcolor="rgba(141,183,232,0.22)",
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Revenue: £%{y:,.2f}"
                        "<extra></extra>"
                    ),
                )

                fig.update_yaxes(
                    title="Revenue (£)",
                    tickformat=",.0f",
                )

                show_chart(fig)

            with right:
                fig = px.bar(
                    sales,
                    x="month",
                    y="total_orders",
                    title="Monthly Orders",
                    color_discrete_sequence=[
                        COLORS["sage_dark"]
                    ],
                    text_auto=True,
                )

                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                fig.update_yaxes(
                    title="Number of Orders",
                    tickformat=",.0f",
                )

                show_chart(fig)

        else:
            st.info(
                "No monthly sales data is available."
            )

    except Exception as exc:
        show_error(
            "Could not load monthly sales",
            exc,
        )

    st.divider()

    st.markdown("## Explore RetailIQ")
    st.caption(
        "Select a module to open its interactive workspace."
    )

    modules = [
        {
            "icon": "🛍️",
            "name": "Retail Analytics",
            "description": (
                "Explore revenue, products, countries "
                "and customer patterns."
            ),
            "color": "#EAF2FC",
        },
        {
            "icon": "🧪",
            "name": "Experimentation",
            "description": (
                "Compare campaigns, conversion "
                "and statistical results."
            ),
            "color": "#EAF5ED",
        },
        {
            "icon": "🎯",
            "name": "Causal ML & Optimization",
            "description": (
                "Explore uplift modeling "
                "and incremental profit."
            ),
            "color": "#F0EBF9",
        },
        {
            "icon": "🤖",
            "name": "Predictive ML",
            "description": (
                "Compare propensity and "
                "customer inactivity models."
            ),
            "color": "#FDF0E7",
        },
        {
            "icon": "📈",
            "name": "Forecasting",
            "description": (
                "Review revenue forecasts "
                "and model accuracy."
            ),
            "color": "#E9F4F4",
        },
        {
            "icon": "💬",
            "name": "AI Assistant",
            "description": (
                "Ask questions about your "
                "RetailIQ findings."
            ),
            "color": "#F8EBF0",
        },
    ]

    for start in range(0, len(modules), 3):
        cols = st.columns(3)

        for offset, col in enumerate(cols):
            index = start + offset

            if index >= len(modules):
                continue

            module = modules[index]

            with col:
                st.markdown(
                    f"""
                    <div class="module-card">
                        <div style="
                            background: {module['color']};
                            width: 46px;
                            height: 46px;
                            border-radius: 12px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 23px;
                        ">
                            {module['icon']}
                        </div>
                        <div class="module-title"
                             style="margin-top: 13px;">
                            {module['name']}
                        </div>
                        <div class="module-description">
                            {module['description']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # IMPORTANT:
                # Use on_click. Do not set navigation_page
                # directly inside an if st.button block.

                st.button(
                    f"Open {module['name']}  →",
                    key=f"home_open_{index}",
                    on_click=navigate_to,
                    args=(module["name"],),
                    width="stretch",
                )


# ============================================================
# RETAIL ANALYTICS
# ============================================================

elif page == "Retail Analytics":

    page_header(
        "Retail & Customer Analytics",
        (
            "Explore sales performance, product contribution "
            "and geographic revenue."
        ),
    )

    try:
        sales = get_data(
            """
            SELECT
                month,
                total_orders,
                units_sold,
                merchandise_revenue,
                identified_customers
            FROM retail.monthly_sales
            ORDER BY month
            """
        )

        products = get_data(
            """
            SELECT
                stock_code,
                description,
                invoice_count,
                units_sold,
                product_revenue
            FROM retail.product_performance
            ORDER BY product_revenue DESC
            LIMIT 10
            """
        )

        countries = get_data(
            """
            SELECT
                country,
                total_orders,
                merchandise_revenue
            FROM retail.country_performance
            ORDER BY merchandise_revenue DESC
            LIMIT 10
            """
        )

        products = clean_numeric_columns(
            products,
            [
                "invoice_count",
                "units_sold",
                "product_revenue",
            ],
        )

        countries = clean_numeric_columns(
            countries,
            [
                "total_orders",
                "merchandise_revenue",
            ],
        )

        # ----------------------------------------------------
        # SALES FILTER
        # ----------------------------------------------------

        if sales.empty:
            st.info(
                "No monthly sales data is available."
            )

        else:
            sales["month"] = pd.to_datetime(
                sales["month"]
            )

            sales = clean_numeric_columns(
                sales,
                [
                    "total_orders",
                    "units_sold",
                    "merchandise_revenue",
                    "identified_customers",
                ],
            )

            start_date = sales["month"].min().date()
            end_date = sales["month"].max().date()

            date_range = st.date_input(
                "Select sales date range",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
            )

            filtered = sales.copy()

            if (
                isinstance(date_range, tuple)
                and len(date_range) == 2
            ):
                start, end = date_range

                filtered = filtered[
                    (filtered["month"].dt.date >= start)
                    & (filtered["month"].dt.date <= end)
                ]

            if not filtered.empty:
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Revenue",
                    (
                        f"£"
                        f"{filtered['merchandise_revenue'].sum():,.0f}"
                    ),
                )

                c2.metric(
                    "Orders",
                    f"{filtered['total_orders'].sum():,.0f}",
                )

                c3.metric(
                    "Units Sold",
                    f"{filtered['units_sold'].sum():,.0f}",
                )

                left, right = st.columns(2)

                with left:
                    fig = px.area(
                        filtered,
                        x="month",
                        y="merchandise_revenue",
                        title="Monthly Revenue",
                        color_discrete_sequence=[
                            COLORS["blue_dark"]
                        ],
                    )

                    fig.update_traces(
                        line=dict(width=3),
                        fillcolor="rgba(141,183,232,0.22)",
                    )

                    fig.update_yaxes(
                        title="Revenue (£)",
                        tickformat=",.0f",
                    )

                    show_chart(fig)

                with right:
                    fig = px.bar(
                        filtered,
                        x="month",
                        y="total_orders",
                        title="Monthly Orders",
                        color_discrete_sequence=[
                            COLORS["sage_dark"]
                        ],
                        text_auto=True,
                    )

                    fig.update_traces(
                        textposition="outside",
                        cliponaxis=False,
                    )

                    fig.update_yaxes(
                        title="Number of Orders",
                        tickformat=",.0f",
                    )

                    show_chart(fig)

        # ----------------------------------------------------
        # PRODUCT PERFORMANCE
        # ----------------------------------------------------

        if not products.empty:
            st.divider()
            st.markdown("## Product Performance")

            products = products.dropna(
                subset=["product_revenue"]
            ).copy()

            products["description"] = (
                products["description"]
                .fillna("Unspecified product")
                .astype(str)
            )

            products = products.sort_values(
                "product_revenue",
                ascending=False,
            )

            if not products.empty:
                product_chart = products.sort_values(
                    "product_revenue",
                    ascending=True,
                ).copy()

                product_chart["display_description"] = (
                    product_chart["description"].apply(
                        lambda value: textwrap.fill(
                            value,
                            width=28,
                            break_long_words=False,
                        )
                    )
                )

                product_height = max(
                    470,
                    45 * len(product_chart) + 150,
                )

                fig = px.bar(
                    product_chart,
                    x="product_revenue",
                    y="display_description",
                    orientation="h",
                    title="Top Products by Revenue",
                    color="product_revenue",
                    color_continuous_scale=[
                        "#DDEBFA",
                        "#477DBD",
                    ],
                    custom_data=[
                        "description",
                        "units_sold",
                        "invoice_count",
                    ],
                )

                maximum_revenue = (
                    product_chart["product_revenue"]
                    .max()
                )

                fig.update_traces(
                    texttemplate="£%{x:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(
                        size=12,
                        color="#26364A",
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Revenue: £%{x:,.2f}<br>"
                        "Units sold: %{customdata[1]:,.0f}<br>"
                        "Invoices: %{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),
                )

                fig.update_xaxes(
                    title="Product Revenue (£)",
                    tickformat=",.0f",
                    range=[
                        0,
                        maximum_revenue * 1.28,
                    ],
                )

                fig.update_yaxes(
                    title="Product",
                    tickfont=dict(
                        size=11,
                        color="#26364A",
                    ),
                    automargin=True,
                )

                show_chart(
                    fig,
                    height=product_height,
                    left_margin=190,
                    right_margin=90,
                    bottom_margin=65,
                )

                # Donut chart with top products plus
                # an aggregated remainder to avoid clutter.

                donut_data = products.head(6).copy()

                remaining_revenue = (
                    products.iloc[6:]["product_revenue"].sum()
                )

                if remaining_revenue > 0:
                    other_row = pd.DataFrame(
                        [{
                            "description": "Other top products",
                            "product_revenue": remaining_revenue,
                        }]
                    )

                    donut_data = pd.concat(
                        [
                            donut_data[
                                [
                                    "description",
                                    "product_revenue",
                                ]
                            ],
                            other_row,
                        ],
                        ignore_index=True,
                    )
                else:
                    donut_data = donut_data[
                        [
                            "description",
                            "product_revenue",
                        ]
                    ]

                fig = px.pie(
                    donut_data,
                    names="description",
                    values="product_revenue",
                    hole=0.55,
                    title="Revenue Share — Top Products",
                    color_discrete_sequence=CHART_COLORS,
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    textfont=dict(
                        size=12,
                        color="#FFFFFF",
                    ),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Revenue: £%{value:,.2f}<br>"
                        "Share: %{percent}"
                        "<extra></extra>"
                    ),
                )

                show_chart(
                    fig,
                    height=440,
                    bottom_margin=70,
                )

                with st.expander(
                    "View complete product data"
                ):
                    st.dataframe(
                        products,
                        width="stretch",
                        hide_index=True,
                    )

        else:
            st.info(
                "No product performance data is available."
            )

        # ----------------------------------------------------
        # COUNTRY PERFORMANCE
        # ----------------------------------------------------

        if not countries.empty:
            st.divider()
            st.markdown("## Country Performance")

            countries = countries.dropna(
                subset=["merchandise_revenue"]
            ).copy()

            countries["country"] = (
                countries["country"]
                .fillna("Unknown")
                .astype(str)
            )

            countries = countries.sort_values(
                "merchandise_revenue",
                ascending=False,
            )

            if not countries.empty:
                country_chart = countries.sort_values(
                    "merchandise_revenue",
                    ascending=True,
                ).copy()

                country_height = max(
                    450,
                    43 * len(country_chart) + 140,
                )

                # Horizontal bars provide sufficient space
                # for country names and revenue labels.

                fig = px.bar(
                    country_chart,
                    x="merchandise_revenue",
                    y="country",
                    orientation="h",
                    title="Revenue by Country",
                    color="merchandise_revenue",
                    color_continuous_scale=[
                        "#E8E0F5",
                        "#8063B4",
                    ],
                    custom_data=["total_orders"],
                )

                maximum_country_revenue = (
                    country_chart["merchandise_revenue"].max()
                )

                fig.update_traces(
                    texttemplate="£%{x:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(
                        size=12,
                        color="#26364A",
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Revenue: £%{x:,.2f}<br>"
                        "Orders: %{customdata[0]:,.0f}"
                        "<extra></extra>"
                    ),
                )

                fig.update_xaxes(
                    title="Merchandise Revenue (£)",
                    tickformat=",.0f",
                    range=[
                        0,
                        maximum_country_revenue * 1.28,
                    ],
                )

                fig.update_yaxes(
                    title="Country",
                    tickfont=dict(
                        size=12,
                        color="#26364A",
                    ),
                    automargin=True,
                )

                show_chart(
                    fig,
                    height=country_height,
                    left_margin=130,
                    right_margin=90,
                    bottom_margin=65,
                )

                # ------------------------------------------------
                # ORDERS VS REVENUE
                # ------------------------------------------------

                scatter_data = countries.dropna(
                    subset=[
                        "total_orders",
                        "merchandise_revenue",
                    ]
                ).copy()

                scatter_data = scatter_data[
                    (scatter_data["total_orders"] > 0)
                    & (scatter_data["merchandise_revenue"] > 0)
                ]

                if not scatter_data.empty:

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=scatter_data["total_orders"],
                            y=scatter_data["merchandise_revenue"],
                            mode="markers",
                            name="Countries",
                            marker=dict(
                                size=(
                                    scatter_data["total_orders"]
                                    / scatter_data["total_orders"].max()
                                    * 38
                                ).clip(lower=10),
                                color=COLORS["blue_dark"],
                                opacity=0.75,
                                line=dict(
                                    color="#FFFFFF",
                                    width=1.5,
                                ),
                                sizemode="diameter",
                            ),
                            customdata=scatter_data[
                                ["country"]
                            ],
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "Total Orders: %{x:,.0f}<br>"
                                "Merchandise Revenue: £%{y:,.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        title="Orders vs Revenue by Country",
                        showlegend=False,
                        hovermode="closest",
                    )

                    fig.update_xaxes(
                        type="log",
                        title="Total Orders (log scale)",
                        tickformat="~s",
                    )

                    fig.update_yaxes(
                        type="log",
                        title="Merchandise Revenue (£, log scale)",
                        tickformat="~s",
                    )

                    show_chart(
                        fig,
                        height=520,
                        left_margin=90,
                        right_margin=55,
                        bottom_margin=85,
                    )

                with st.expander(
                    "View complete country data"
                ):
                    st.dataframe(
                        countries,
                        width="stretch",
                        hide_index=True,
                    )

        else:
            st.info(
                "No country performance data is available."
            )

    except Exception as exc:
        show_error(
            "Retail analytics error",
            exc,
        )


# ============================================================
# EXPERIMENTATION
# ============================================================

elif page == "Experimentation":

    page_header(
        "Campaign Experimentation",
        (
            "Compare campaign outcomes and review "
            "statistical experiment quality."
        ),
    )

    try:
        summary = get_data(
            """
            SELECT
                segment,
                COUNT(*) AS customers,
                SUM(visit) AS visits,
                SUM(conversion) AS conversions,
                AVG(spend) AS revenue_per_customer
            FROM hillstrom.campaign_data
            GROUP BY segment
            ORDER BY segment
            """
        )

        if summary.empty:
            st.info(
                "No campaign data is available."
            )

        else:
            summary = clean_numeric_columns(
                summary,
                [
                    "customers",
                    "visits",
                    "conversions",
                    "revenue_per_customer",
                ],
            )

            summary["conversion_rate"] = (
                summary["conversions"]
                / summary["customers"]
                * 100
            )

            summary["visit_rate"] = (
                summary["visits"]
                / summary["customers"]
                * 100
            )

            metric_options = {
                "Conversion Rate": "conversion_rate",
                "Visit Rate": "visit_rate",
                "Revenue per Customer": "revenue_per_customer",
            }

            selected_metric = st.selectbox(
                "Campaign performance metric",
                list(metric_options.keys()),
            )

            metric_column = metric_options[
                selected_metric
            ]

            fig = px.bar(
                summary,
                x="segment",
                y=metric_column,
                color="segment",
                title=selected_metric + " by Campaign",
                color_discrete_sequence=CHART_COLORS,
                text_auto=".3f",
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_yaxes(
                title=selected_metric,
            )

            show_chart(fig)

            left, right = st.columns(2)

            with left:
                fig = px.pie(
                    summary,
                    names="segment",
                    values="customers",
                    hole=0.55,
                    title="Campaign Customer Allocation",
                    color_discrete_sequence=CHART_COLORS,
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                    textfont=dict(
                        size=12,
                        color="#FFFFFF",
                    ),
                )

                show_chart(fig)

            with right:
                fig = px.bar(
                    summary,
                    x="segment",
                    y=["visit_rate", "conversion_rate"],
                    barmode="group",
                    title="Visit Rate vs Conversion Rate",
                    color_discrete_sequence=[
                        COLORS["blue_dark"],
                        COLORS["sage_dark"],
                    ],
                    labels={
                        "value": "Rate (%)",
                        "variable": "Metric",
                        "segment": "Campaign",
                    },
                    text_auto=".2f",
                )

                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                show_chart(fig)

            with st.expander(
                "View campaign summary"
            ):
                st.dataframe(
                    summary,
                    width="stretch",
                    hide_index=True,
                )

            st.divider()

            st.markdown("## Statistical Significance")

            conversion_tests = pd.DataFrame({
                "Comparison": [
                    "Men's E-Mail vs No E-Mail",
                    "Women's E-Mail vs No E-Mail",
                    "Men's E-Mail vs Women's E-Mail",
                ],
                "Holm-adjusted result": [
                    "Significant",
                    "Significant",
                    "Significant",
                ],
            })

            revenue_tests = conversion_tests.copy()

            left, right = st.columns(2)

            with left:
                st.markdown(
                    "**Conversion comparisons**"
                )

                st.dataframe(
                    conversion_tests,
                    width="stretch",
                    hide_index=True,
                )

            with right:
                st.markdown(
                    "**Revenue comparisons**"
                )

                st.dataframe(
                    revenue_tests,
                    width="stretch",
                    hide_index=True,
                )

            st.caption(
                "All three pairwise conversion comparisons and "
                "revenue comparisons were significant after "
                "Holm correction. Exact pairwise p-values "
                "are not displayed."
            )

            st.divider()

            st.markdown("## Experiment Quality Check")

            left, right = st.columns([1, 1.5])

            with left:
                show_chart(
                    make_gauge(
                        0.9037,
                        "SRM p-value",
                        threshold=0.05,
                    )
                )

            with right:
                st.metric(
                    "Expected Allocation",
                    "1:1:1",
                )

                st.metric(
                    "SRM Detected",
                    "No",
                )

                st.success(
                    "No evidence of sample-ratio mismatch "
                    "was detected under the assumed equal "
                    "1:1:1 allocation."
                )

                st.caption(
                    "The gauge shows the SRM p-value and "
                    "the 0.05 reference threshold, not "
                    "campaign performance."
                )

    except Exception as exc:
        show_error(
            "Campaign analytics error",
            exc,
        )


# ============================================================
# CAUSAL ML & OPTIMIZATION
# ============================================================

elif page == "Causal ML & Optimization":

    page_header(
        "Causal ML & Profit Optimization",
        (
            "Evaluate campaign targeting strategies "
            "using estimated incremental profit."
        ),
    )

    strategies = pd.DataFrame({
        "Strategy": [
            "No campaign",
            "Random 30% allocation",
            "Uplift-based 30% allocation",
        ],
        "Incremental Profit": [
            0.00,
            227.95,
            167.94,
        ],
    })

    intervals = pd.DataFrame({
        "Strategy": [
            "Random 30% allocation",
            "Uplift-based 30% allocation",
        ],
        "Lower Bound": [
            142.23,
            -1.53,
        ],
        "Upper Bound": [
            313.86,
            337.45,
        ],
    })

    c1, c2 = st.columns(2)

    c1.metric(
        "Highest Estimated Profit",
        "£227.95",
        "Random 30% per 1,000 customers",
    )

    c2.metric(
        "Uplift-Based Profit",
        "£167.94",
        "Per 1,000 customers",
    )

    st.markdown(
        "## Incremental Profit by Strategy"
    )

    fig = px.bar(
        strategies,
        x="Strategy",
        y="Incremental Profit",
        color="Strategy",
        title=(
            "Estimated Incremental Profit "
            "(£ per 1,000 Customers)"
        ),
        color_discrete_sequence=CHART_COLORS,
        text_auto=".2f",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_yaxes(
        title="Incremental Profit (£)",
    )

    show_chart(
        fig,
        height=430,
        bottom_margin=100,
    )

    st.markdown(
        "## Bootstrap Confidence Intervals"
    )

    fig = go.Figure()

    for _, row in intervals.iterrows():
        midpoint = (
            row["Lower Bound"]
            + row["Upper Bound"]
        ) / 2

        fig.add_trace(
            go.Scatter(
                x=[midpoint],
                y=[row["Strategy"]],
                mode="markers",
                marker=dict(
                    size=13,
                    color=COLORS["blue_dark"],
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        row["Upper Bound"] - midpoint
                    ],
                    arrayminus=[
                        midpoint - row["Lower Bound"]
                    ],
                    thickness=3,
                    width=9,
                ),
                name=row["Strategy"],
                showlegend=False,
                hovertemplate=(
                    "%{y}<br>"
                    "Estimated profit: £%{x:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#D97757",
        annotation_text="Zero profit",
        annotation_font_color="#9A4D35",
    )

    fig.update_layout(
        title="Profit Estimates and Bootstrap Intervals",
        xaxis_title=(
            "Incremental Profit (£ per 1,000 Customers)"
        ),
        yaxis_title="",
    )

    show_chart(
        fig,
        height=360,
        left_margin=180,
    )

    st.dataframe(
        intervals,
        width="stretch",
        hide_index=True,
    )

    st.markdown("## Business Interpretation")

    st.write(
        "Random 30% allocation generated the highest "
        "estimated incremental profit: £227.95 per "
        "1,000 customers, compared with £167.94 "
        "for uplift-based allocation."
    )

    st.warning(
        "The uplift-based confidence interval ranges "
        "from -£1.53 to £337.45 per 1,000 customers. "
        "It includes zero."
    )

    st.info(
        "The £0.01 email cost is illustrative. "
        "The evaluation uses five-fold cross-fitting. "
        "Bootstrap intervals hold policy decisions fixed "
        "and do not include model-retraining uncertainty."
    )


# ============================================================
# PREDICTIVE ML
# ============================================================

elif page == "Predictive ML":

    page_header(
        "Predictive Machine Learning",
        (
            "Compare purchase propensity and "
            "customer inactivity models."
        ),
    )

    propensity = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
        "ROC-AUC": [
            0.7202,
            0.7152,
            0.6900,
        ],
        "PR-AUC": [
            0.6599,
            0.6512,
            0.6258,
        ],
        "Precision": [
            0.6269,
            0.6103,
            0.7245,
        ],
        "Recall": [
            0.5221,
            0.5631,
            0.2800,
        ],
        "F1": [
            0.5697,
            0.5857,
            0.4039,
        ],
    })

    churn = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
        "ROC-AUC": [
            0.7231,
            0.7279,
            0.7161,
        ],
        "PR-AUC": [
            0.5664,
            0.5680,
            0.5513,
        ],
        "Precision": [
            0.5126,
            0.5239,
            0.5279,
        ],
        "Recall": [
            0.7836,
            0.7413,
            0.7554,
        ],
        "F1": [
            0.6198,
            0.6139,
            0.6215,
        ],
    })

    task = st.radio(
        "Prediction task",
        [
            "Purchase Propensity",
            "Customer Inactivity",
        ],
        horizontal=True,
    )

    selected = (
        propensity
        if task == "Purchase Propensity"
        else churn
    )

    st.markdown(
        f"## {task} Model Comparison"
    )

    metric = st.selectbox(
        "Choose evaluation metric",
        [
            "ROC-AUC",
            "PR-AUC",
            "Precision",
            "Recall",
            "F1",
        ],
    )

    # --------------------------------------------------------
    # SELECTED METRIC COMPARISON
    # --------------------------------------------------------

    fig = px.bar(
        selected,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} by Model",
        color_discrete_sequence=CHART_COLORS,
        text_auto=".4f",
    )

    fig.update_yaxes(
        range=[0, 1],
        title=metric,
        tickformat=".0%",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    show_chart(
        fig,
        height=420,
        bottom_margin=105,
    )

    # --------------------------------------------------------
    # PRECISION VS RECALL
    # --------------------------------------------------------
    #
    # The previous scatter chart placed very similar
    # model points on top of one another.
    #
    # This grouped bar chart displays Precision and Recall
    # separately for each model, so all six values remain
    # visible and comparable.

    st.markdown("## Precision vs Recall Comparison")

    precision_recall = selected[
        [
            "Model",
            "Precision",
            "Recall",
        ]
    ].melt(
        id_vars="Model",
        value_vars=[
            "Precision",
            "Recall",
        ],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        precision_recall,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Precision and Recall by Model",
        color_discrete_map={
            "Precision": COLORS["blue_dark"],
            "Recall": COLORS["sage_dark"],
        },
        text_auto=".3f",
        category_orders={
            "Metric": [
                "Precision",
                "Recall",
            ]
        },
    )

    fig.update_yaxes(
        range=[0, 1],
        title="Score",
        tickformat=".0%",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    show_chart(
        fig,
        height=440,
        bottom_margin=110,
    )

    # --------------------------------------------------------
    # MODEL METRIC HEATMAP
    # --------------------------------------------------------

    st.markdown("## Model Evaluation Heatmap")

    heatmap = selected.set_index("Model")[
        [
            "ROC-AUC",
            "PR-AUC",
            "Precision",
            "Recall",
            "F1",
        ]
    ]

    fig = px.imshow(
        heatmap,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale=[
            "#F5F7FB",
            "#C8B8E8",
            "#477DBD",
        ],
        title="Model Evaluation Metrics",
        zmin=0,
        zmax=1,
    )

    fig.update_traces(
        textfont=dict(
            size=13,
            color="#172B4D",
        )
    )

    fig.update_xaxes(
        side="bottom",
        tickfont=dict(
            size=12,
            color="#26364A",
        ),
    )

    fig.update_yaxes(
        tickfont=dict(
            size=12,
            color="#26364A",
        ),
    )

    show_chart(
        fig,
        height=350,
        left_margin=155,
        bottom_margin=90,
    )

    with st.expander(
        "View complete model evaluation table"
    ):
        st.dataframe(
            selected,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------------

    if task == "Purchase Propensity":

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Best ROC-AUC",
            "0.7202",
            "Logistic Regression",
        )

        c2.metric(
            "Best PR-AUC",
            "0.6599",
            "Logistic Regression",
        )

        c3.metric(
            "Best Precision",
            "0.7245",
            "XGBoost",
        )

        st.write(
            "Logistic Regression achieved the highest "
            "ROC-AUC and PR-AUC. XGBoost achieved the "
            "highest precision but had lower recall."
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Best ROC-AUC",
            "0.7279",
            "Random Forest",
        )

        c2.metric(
            "Best PR-AUC",
            "0.5680",
            "Random Forest",
        )

        c3.metric(
            "Best F1",
            "0.6215",
            "XGBoost",
        )

        st.write(
            "Random Forest achieved the highest "
            "ROC-AUC and PR-AUC, while XGBoost "
            "achieved the highest F1 score."
        )

    st.warning(
        "These are preliminary evaluations. Propensity "
        "snapshots overlap in historical periods and "
        "include customer overlap, so they are not "
        "independent holdout performance. The inactivity "
        "evaluation uses one train/test cutoff pair and "
        "requires additional temporal validation."
    )


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    page_header(
        "Revenue Forecasting",
        (
            "Compare forecasting models and inspect "
            "historical predictions."
        ),
    )

    forecast_metrics = pd.DataFrame({
        "Model": [
            "Seasonal Naive",
            "Holt-Winters",
        ],
        "MAE (£)": [
            147508.75,
            274734.11,
        ],
        "RMSE (£)": [
            255044.70,
            395445.27,
        ],
        "MAPE (%)": [
            21.53,
            23.10,
        ],
    })

    actual_forecast = pd.DataFrame({
        "Month": pd.to_datetime([
            "2011-07-01",
            "2011-08-01",
            "2011-09-01",
            "2011-10-01",
            "2011-11-01",
            "2011-12-01",
        ]),
        "Actual Revenue": [
            689947.74,
            726005.16,
            1031430.36,
            1107449.54,
            1458925.53,
            615716.60,
        ],
        "Holt-Winters Forecast": [
            723839.43,
            708251.98,
            692664.53,
            677077.08,
            661489.63,
            645902.18,
        ],
    })

    st.markdown(
        "## Forecast Model Comparison"
    )

    metric = st.selectbox(
        "Forecast evaluation metric",
        [
            "MAE (£)",
            "RMSE (£)",
            "MAPE (%)",
        ],
    )

    fig = px.bar(
        forecast_metrics,
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison — {metric}",
        color_discrete_sequence=CHART_COLORS,
        text_auto=",.2f",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_yaxes(
        title=metric,
    )

    show_chart(fig)

    st.dataframe(
        forecast_metrics,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.markdown(
        "## Actual vs Holt-Winters Forecast"
    )

    view = st.radio(
        "Chart view",
        [
            "Revenue trend",
            "Forecast error",
        ],
        horizontal=True,
    )

    if view == "Revenue trend":

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=actual_forecast["Month"],
                y=actual_forecast["Actual Revenue"],
                mode="lines+markers",
                name="Actual Revenue",
                line=dict(
                    color=COLORS["blue_dark"],
                    width=3,
                ),
                marker=dict(
                    size=9,
                ),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Actual Revenue: £%{y:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=actual_forecast["Month"],
                y=actual_forecast[
                    "Holt-Winters Forecast"
                ],
                mode="lines+markers",
                name="Holt-Winters Forecast",
                line=dict(
                    color=COLORS["peach_dark"],
                    width=3,
                    dash="dash",
                ),
                marker=dict(
                    size=9,
                ),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Forecast: £%{y:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=(
                "Actual Revenue vs Holt-Winters Forecast"
            ),
            xaxis_title="Month",
            yaxis_title="Revenue (£)",
            hovermode="x unified",
        )

        fig.update_yaxes(
            tickformat=",.0f",
        )

    else:

        error_data = actual_forecast.copy()

        error_data["Forecast Error"] = (
            error_data["Actual Revenue"]
            - error_data["Holt-Winters Forecast"]
        )

        fig = px.bar(
            error_data,
            x="Month",
            y="Forecast Error",
            title="Actual Minus Forecast Revenue",
            color="Forecast Error",
            color_continuous_scale=[
                "#E8B7C8",
                "#F5F7FB",
                "#A9CEB3",
            ],
            text_auto=",.0f",
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#718096",
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_yaxes(
            title="Forecast Error (£)",
            tickformat=",.0f",
        )

    show_chart(
        fig,
        height=440,
        bottom_margin=90,
    )

    with st.expander(
        "View forecast data"
    ):
        st.dataframe(
            actual_forecast,
            width="stretch",
            hide_index=True,
        )

    st.success(
        "Seasonal Naive performed better than "
        "Holt-Winters on MAE, RMSE and MAPE "
        "in this evaluation."
    )

    st.caption(
        "The evaluation uses a six-month historical "
        "holdout. The available training history was "
        "insufficient for a reliable full seasonal "
        "Holt-Winters model."
    )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "AI Assistant":

    page_header(
        "RetailIQ AI Decision Assistant",
        (
            "Ask questions about campaign experiments, "
            "models, forecasting and business findings."
        ),
    )

    st.markdown(
        """
        <div class="hero">
            <div style="font-size: 25px;">💬</div>
            <div class="hero-title" style="font-size: 23px;">
                Your Retail Analytics Assistant
            </div>
            <div class="hero-subtitle">
                Ask questions and receive answers grounded
                in the RetailIQ decision context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggested_questions = [
        (
            "Which campaign allocation strategy generated "
            "the highest estimated incremental profit?"
        ),
        (
            "Compare the campaign conversion rates "
            "and explain the SRM result."
        ),
        (
            "Which forecasting model performed best?"
        ),
        (
            "Compare the purchase propensity models."
        ),
    ]

    st.markdown("## Suggested Questions")

    for index, suggestion in enumerate(
        suggested_questions
    ):
        st.button(
            suggestion,
            key=f"suggested_question_{index}",
            on_click=load_suggested_question,
            args=(suggestion,),
            width="stretch",
        )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Ask RetailIQ about your analysis..."
        ),
        height=130,
        key="assistant_input",
    )

    if st.button(
        "Ask RetailIQ  →",
        type="primary",
        key="ask_retailiq",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:
                with st.spinner(
                    "Analyzing RetailIQ results..."
                ):

                    response = requests.post(
                        ASSISTANT_URL,
                        json={
                            "question": question,
                        },
                        timeout=180,
                    )

                    response.raise_for_status()

                    response_data = response.json()

                    answer = response_data["answer"]

                st.markdown("## Answer")

                st.markdown(answer)

                st.caption(
                    "Answers use the RetailIQ project "
                    "decision context. Verify important "
                    "business decisions against the "
                    "underlying analysis."
                )

            except requests.RequestException as exc:

                st.error(
                    f"Assistant connection failed: {exc}"
                )

            except (KeyError, ValueError):

                st.error(
                    "The assistant returned an invalid response."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        RetailIQ · Retail Experimentation, Causal ML
        & AI Decision Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)