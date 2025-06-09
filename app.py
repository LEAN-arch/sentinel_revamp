# sentinel_project_root/app.py
#
# PLATINUM STANDARD - Main Application Entry Point
# This file serves as the landing page for the Sentinel Health Co-Pilot.
# It is designed to be clean, robust, and extensible, dynamically discovering
# and linking to all available dashboards.

import streamlit as st
import logging
from pathlib import Path
import re

# --- Core Application Imports ---
# The application's configuration is centralized in the 'settings' object.
try:
    from config.settings import settings
except ImportError:
    st.error("FATAL ERROR: The application's configuration `config.settings` could not be loaded. Ensure the file exists and is correct.")
    st.stop()
except Exception as e:
    st.error(f"An unhandled exception occurred during configuration import: {e}")
    st.stop()

# --- Logging Configuration ---
# Set up global logging based on the level defined in settings.
logging.basicConfig(
    level=settings.app.log_level,
    format=settings.app.log_format,
    datefmt=settings.app.log_date_format,
    force=True  # Override any existing handlers
)
logger = logging.getLogger(__name__)

# --- Page & Theme Configuration ---
# This block sets up the fundamental look and feel of the Streamlit application.
st.set_page_config(
    page_title=f"{settings.app.name} - System Overview",
    page_icon=str(settings.app_logo_small_path) if settings.app_logo_small_path.exists() else "🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.app.support_contact}?subject=Help Request - {settings.app.name}",
        "Report a bug": f"mailto:{settings.app.support_contact}?subject=Bug Report - {settings.app.name} v{settings.app.version}",
        "About": f"### {settings.app.name} (v{settings.app.version})\n\n{settings.app_footer_text}"
    }
)

# --- Apply Global CSS ---
@st.cache_resource
def load_css(path: Path):
    """Loads a CSS file and injects it into the Streamlit app."""
    if not path.is_file():
        logger.warning(f"CSS file not found at {path}. Skipping custom styles.")
        return
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logger.debug(f"Successfully loaded global CSS from {path}.")
    except Exception as e:
        logger.error(f"Error loading CSS from {path}: {e}", exc_info=True)

load_css(settings.style_css_path)

# --- Main Application Header ---
# This consistent header appears on the main page.
header_cols = st.columns([0.1, 0.9])
with header_cols[0]:
    if settings.app_logo_large_path.exists():
        st.image(str(settings.app_logo_large_path), width=90)
    else:
        st.markdown("### 🌍")

with header_cols[1]:
    st.title(settings.app.name)
    st.subheader("Edge-First Health Intelligence & Decision Support System")
st.divider()


# --- Introduction & Core Principles ---
st.markdown(f"""
## Welcome to the {settings.app.name}
Sentinel is a platinum-grade, **edge-first health intelligence ecosystem** designed for maximum clinical
and operational actionability in resource-limited or high-risk environments. It transforms diverse data
streams into life-saving, workflow-integrated decisions, with or without internet connectivity.
""")

st.markdown("#### Core Design Philosophy:")
core_principles = [
    ("⚡ **Edge-First, Offline Capable**", "Critical analytics and decision support run directly on low-power devices at the point of care, ensuring function without constant connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight and visualization is engineered to prompt a specific, high-value decision or intervention relevant to frontline workflows."),
    ("🤝 **Human-Centered & Resilient**", "Interfaces are optimized for high-stress users, prioritizing clarity and speed. The system is designed for robust data sync in intermittent networks."),
    ("📈 **Predictive & Inferential**", "The system moves beyond descriptive reporting to forecast supply needs, stratify patient risk, and identify statistically significant epidemiological trends.")
]

for icon, title, desc in [(*p, "") for i, p in enumerate(core_principles)]: # Adapted to fit your structure
    with st.expander(f"{icon} {title}", expanded=False):
        st.markdown(desc if desc else title) # Displaying title as content if desc is empty


# --- Dynamic Dashboard Navigation ---
st.markdown("---")
st.header("Explore Role-Specific Dashboards")
st.markdown(
    "Select a dashboard below to view perspectives tailored to different operational tiers. "
    "These views demonstrate the aggregated intelligence available to supervisors and managers."
)

PAGES_DIR = settings.directories.root / "pages"
DASHBOARD_PAGES = sorted(PAGES_DIR.glob("[0-9]*.py"))

# Create columns for navigation cards.
if DASHBOARD_PAGES:
    cols = st.columns(len(DASHBOARD_PAGES))
    for i, page_path in enumerate(DASHBOARD_PAGES):
        # Derive title from filename (e.g., '01_field_operations.py' -> 'Field Operations')
        page_stem = page_path.stem
        cleaned_title = re.sub(r"^\d+_?", "", page_stem).replace("_", " ").title()
        icon = {"Field": "🧑‍⚕️", "Clinic": "🏥", "District": "🗺️", "Population": "📊"}.get(cleaned_title.split()[0], "📈")

        with cols[i]:
            with st.container(border=True):
                st.subheader(f"{icon} {cleaned_title}")
                st.page_link(str(page_path), label=f"Open {cleaned_title} View", use_container_width=True)
else:
    st.warning("No dashboard pages found in the `pages` directory.")

st.divider()

# --- Sidebar Content ---
st.sidebar.title(f"{settings.app.name}")
st.sidebar.markdown(f"`Version {settings.app.version}`")
st.sidebar.info(
    "This web application simulates high-level dashboards for strategic oversight. "
    "The primary interface for frontline workers is a dedicated offline-first mobile application."
)
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.app.organization_name}**")
st.sidebar.markdown(f"Support: <a href='mailto:{settings.app.support_contact}'>{settings.app.support_contact}</a>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption(settings.app_footer_text)

logger.info(f"System overview page loaded for {settings.app.name} v{settings.app.version}.")
