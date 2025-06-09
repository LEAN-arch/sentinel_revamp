# sentinel_project_root/app.py
# Final, Corrected, and Validated Main Application Entry Point

import streamlit as st
import logging
from pathlib import Path
import re

try:
    from config.settings import settings
except Exception as e:
    st.set_page_config(page_title="Fatal Error", layout="centered")
    st.error(f"CRITICAL ERROR: The application's configuration could not be loaded.\n\nDetails: {e}")
    st.stop()

# --- THE FIX: Logging config now correctly references top-level settings attributes ---
logging.basicConfig(
    level=settings.log_level,
    format=settings.log_format,
    datefmt=settings.log_date_format,
    force=True
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{settings.app.name} - System Overview",
    page_icon=str(settings.app_logo_large_path.parent / "sentinel_logo_small.png") if (settings.app_logo_large_path.parent / "sentinel_logo_small.png").exists() else "🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.app.support_contact}",
        "Report a bug": f"mailto:{settings.app.support_contact}",
        "About": f"### {settings.app.name} (v{settings.app.version})\n\n{settings.app_footer_text}"
    }
)

@st.cache_resource
def load_css(path: Path):
    if path and path.is_file():
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(settings.style_css_path)

header_cols = st.columns([0.1, 0.9])
with header_cols[0]:
    if settings.app_logo_large_path and settings.app_logo_large_path.exists():
        st.image(str(settings.app_logo_large_path), width=90)
    else:
        st.markdown("### 🌍")
with header_cols[1]:
    st.title(settings.app.name)
    st.subheader("Edge-First Health Intelligence & Decision Support System")
st.divider()

st.markdown(f"## Welcome to the {settings.app.name}")
st.markdown("#### Core Design Philosophy:")
core_principles = [
    ("⚡ **Edge-First, Offline Capable**", "Critical analytics run on low-power devices, ensuring function without constant connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight is engineered to prompt a specific, high-value decision or intervention."),
    ("🤝 **Human-Centered & Resilient**", "Interfaces are optimized for high-stress users. The system is designed for robust data sync in intermittent networks."),
    ("📈 **Predictive & Inferential**", "The system moves beyond reporting to forecast needs, stratify risk, and identify significant trends.")
]
for icon, title, desc in [(*p, "") for p in core_principles]:
    with st.expander(f"{icon} {title}", expanded=False):
        st.markdown(desc if desc else title.split('**', 2)[-1].strip())

st.markdown("---")
st.header("Explore Role-Specific Dashboards")

PAGES_DIR = settings.directories.root / "pages"
DASHBOARD_PAGES = sorted(PAGES_DIR.glob("[0-9]*.py"))

if DASHBOARD_PAGES:
    cols = st.columns(len(DASHBOARD_PAGES) if DASHBOARD_PAGES else 1)
    for i, page_path in enumerate(DASHBOARD_PAGES):
        cleaned_title = re.sub(r"^\d+_?", "", page_path.stem).replace("_", " ").title()
        icon = {"Field": "🧑‍⚕️", "Clinic": "🏥", "District": "🗺️", "Population": "📊"}.get(cleaned_title.split()[0], "📈")
        with cols[i]:
            with st.container(border=True):
                st.subheader(f"{icon} {cleaned_title}")
                st.page_link(str(page_path), label=f"Open {cleaned_title} View", use_container_width=True)
else:
    st.warning("No dashboard pages found in the `pages` directory.")

st.sidebar.title(f"{settings.app.name}")
st.sidebar.markdown(f"`Version {settings.app.version}`")
st.sidebar.info("This web application simulates high-level dashboards for strategic oversight.")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.app.organization_name}**")
st.sidebar.markdown(f"Support: <a href='mailto:{settings.app.support_contact}'>{settings.app.support_contact}</a>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption(settings.app_footer_text)
