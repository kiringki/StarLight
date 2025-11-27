# st.set_page_config 호출하고, 쿼리파라미터/사이드바/라우팅을 처리


import streamlit as st

from analysis_page import render_analysis_page
from starlight_core import (
    init_session_state,
    render_condition_summary,
)
from starlight_pages import (
    render_overview_page,
    render_data_model_page
)


# ============================
# 페이지 기본 설정
# ============================
st.set_page_config(
    page_title="STARLIGHT",
    page_icon=":stars:",
    layout="wide",
)

# 세션 기본값 설정
init_session_state()

# ============================
# URL 쿼리 파라미터 기반 페이지 체크
# ============================
ALL_PAGES = ["실시간 지도 분석", "이미지 분석", "분석 결과"]
SIDEBAR_PAGES = ["실시간 지도 분석", "이미지 분석"]

query_params = st.query_params
current_qp_page = query_params.get("page", None)

# 팝업으로 열린 "분석 결과" 창이면 쿼리 파라미터 → session_state 반영
if current_qp_page == "분석 결과":
    eq = query_params.get("equipment", None)
    radius = query_params.get("radius", None)
    place = query_params.get("place", None)
    lon = query_params.get("lon", None)
    lat = query_params.get("lat", None)

    if eq:
        st.session_state.final_equipment = eq
    if radius:
        try:
            st.session_state.radius_km = float(radius)
        except ValueError:
            pass
    if place:
        st.session_state.place_name = place
    if lon:
        try:
            st.session_state.coord_lon = float(lon)
        except ValueError:
            pass
    if lat:
        try:
            st.session_state.coord_lat = float(lat)
        except ValueError:
            pass

# ============================
# 좌측 사이드바 - 전역 메뉴
# ============================
if current_qp_page != "분석 결과":
    st.sidebar.markdown("# StarLight🌠")
    st.sidebar.markdown(
        "<span style='font-size:0.85rem; color:#6b7280;'>"
        "위성영상을 활용한 <br> 軍 장비 기동가능성 평가 AI모델"
        "</span>",
        unsafe_allow_html=True,
    )

    selected_label = st.sidebar.radio(
        " ",
        SIDEBAR_PAGES,
        key="sidebar_menu",
    )
    selected_page = selected_label
else:
    selected_page = "분석 결과"

# ============================
# 페이지 라우팅
# ============================
if selected_page == "실시간 지도 분석":
    render_overview_page()
elif selected_page == "이미지 분석":
    render_data_model_page()
elif selected_page == "분석 결과":
    render_analysis_page(render_condition_summary)
