import os
from typing import Literal

import dotenv
import polars as pl
import streamlit as st
from loguru import logger

# Replace this:
# from ..constants.tables import Table
# from ..router import SECTIONS
# from ..state.feedback import show_success_message
# from ..state.tables import is_loaded, loaded_tables
# from utils.persistence import user_state
# from utils.persistence.save_load import loader, session_available

# With this:
from Project_Ensemble_analytics.constants.tables import Table
from Project_Ensemble_analytics.router import SECTIONS
from Project_Ensemble_analytics.state.feedback import show_success_message
from Project_Ensemble_analytics.state.tables import is_loaded, loaded_tables
from Project_Ensemble_analytics.utils.persistence import user_state
from Project_Ensemble_analytics.utils.persistence.save_load import loader, session_available


@st.cache_data
def page_switcher(loaded_tables: list[str] | None = None) -> None:
    logger.debug(id(user_state))
    logger.debug(list(user_state.keys()))

    for section in SECTIONS:
        st.caption(section.title)
        for page in section.pages:
            required_tables = set(page.required_tables or []) | set(
                section.required_tables or []
            )

            missing_tables = (
                [t for t in required_tables if t not in loaded_tables]
                if loaded_tables
                else []
            )

            st.page_link(
                f"./pages/{page.path}.py",
                label=page.title,
                icon=page.icon,
                disabled=len(missing_tables) > 0,
                help=(
                    f"Missing {', '.join(missing_tables)} data."
                    if missing_tables
                    else None
                ),
            )


def get_urls() -> dict[str, str]:
    dotenv.load_dotenv()

    try:
        host = os.environ["INGRESS_HOST_NAME"].strip()
    except KeyError:
        return {
            "get help": "https://youtu.be/pxw-5qfJ1dk",
            "about": "https://youtu.be/pxw-5qfJ1dk",
            "report a bug": "https://youtu.be/pxw-5qfJ1dk",
        }

    # StreamlitAPIException: We only accept the keys:
    # "Get help", "Report a bug", and "About"
    # ("sign out" is not a valid key.)
    # tf...

    return {
        "get help": f"https://{host}/oauth2/sign_out",
        "about": f"https://{host}/oauth2/sign_in",
        "report a bug": f"https://{host}/oauth2/user_info",
    }


def init_page(
    page_title: str | None = None,
    *,
    page_icon: str | None = None,
    layout: Literal["wide", "centered"] = "wide",
    required_tables: list[Table] | None = None,
) -> None:
    if not pl.using_string_cache():
        logger.info("Enabling string cache.")
        pl.enable_string_cache()

    logger.debug(
        "Initializing page"
        f" {page_title} {page_icon} {layout} {required_tables}"
    )
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon or ":material/trending_down:",
        layout=layout,
        menu_items=get_urls(),  # type: ignore[arg-type]
    )

    if (
        not user_state.get("resume_session_ignored", False)
        and session_available()
    ):
        st.info("A saved session is available. Resume?")
        c1, c2 = st.columns(2)
        if c1.button("Yes.", use_container_width=True):
            status = st.status("Loading previous session")
            loader()
            user_state["resume_session_ignored"] = True
            status.success("Session resumed")
            st.rerun()

        if c2.button("No.", use_container_width=True):
            user_state["resume_session_ignored"] = True
            st.rerun()

        st.stop()

    st.title(page_title or "Well potential tool")

    with st.sidebar:
        st.logo(
            "data/logo.png",
            size="large"
        )
        page_switcher(loaded_tables())
        st.divider()

    st.markdown(
        """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )

    show_success_message()

    if required_tables:
        missing = [t for t in required_tables if not is_loaded(t)]
        if missing:
            st.error(f"Missing {', '.join(missing)} data.")
            st.stop()
