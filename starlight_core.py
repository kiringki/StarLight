# 공통 유틸 + 세션 상태 초기화 + 장비 DB + 팝업 + 요약


import os
import json

import streamlit as st
from streamlit.components.v1 import html
from map_api import TACTICAL_SCALE_PRESETS



# ============================
# 사용자 정의 장비 스펙 JSON 설정
# ============================
USER_SPEC_PATH = "user_equipment_specs.json"


def load_user_equipment_specs() -> dict:
    """사용자 정의 장비 스펙 JSON에서 불러오기"""
    if os.path.exists(USER_SPEC_PATH):
        try:
            with open(USER_SPEC_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def save_user_equipment_spec(name: str, length: float, width: float, weight: float):
    """사용자 정의 장비 스펙 저장/업데이트"""
    data = load_user_equipment_specs()
    data[name] = {
        "length": length,
        "width": width,
        "weight": weight,
    }
    with open(USER_SPEC_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================
# 초기 session_state 설정
# ============================
def init_session_state():
    """앱에서 공통으로 사용하는 session_state 기본값 설정"""

    if "place_name" not in st.session_state:
        st.session_state.place_name = ""

    if "coord_lon" not in st.session_state:
        # 기본값 (입력 안 한 상태를 0.0, 0.0으로 취급)
        st.session_state.coord_lon = 0.0
    if "coord_lat" not in st.session_state:
        st.session_state.coord_lat = 0.0

    if "radius_km" not in st.session_state:
        # 전술지도 축척에서 자동으로 세팅해 줄 값 (초기값은 3km 정도)
        st.session_state.radius_km = 3.0

    if "show_current_location" not in st.session_state:
        st.session_state.show_current_location = False

    if "final_equipment" not in st.session_state:
        st.session_state.final_equipment = "K-1 전차"

    if "equip_length" not in st.session_state:
        st.session_state.equip_length = 0.0

    if "equip_width" not in st.session_state:
        st.session_state.equip_width = 0.0

    if "equip_weight" not in st.session_state:
        st.session_state.equip_weight = 0.0

    if "selected_zoom" not in st.session_state:
        # 전술지도 축척 선택 전 기본 zoom
        st.session_state.selected_zoom = 16

    # ===== 경로 분석 공통 상태 (실시간 / 이미지 모드 둘 다 사용) =====
    if "route_start" not in st.session_state:
        st.session_state.route_start = None

    if "route_end" not in st.session_state:
        st.session_state.route_end = None

    if "obstacles_px" not in st.session_state:
        st.session_state.obstacles_px = []

    # 해상도(m/px) – 실시간 지도 / 이미지 분석 모두 공통으로 사용
    if "res_m_per_px" not in st.session_state:
        st.session_state.res_m_per_px = 0.55


# ============================
# 분석 결과 페이지를 새 탭(팝업)으로 여는 함수
# ============================
def open_analysis_popup():
    """현재 session_state의 조건을 쿼리파라미터로 넘겨 분석결과 페이지를 새 탭으로 연다."""
    final_equipment = st.session_state.get("final_equipment", "")
    radius_km = st.session_state.get("radius_km", "")
    place_name = st.session_state.get("place_name", "")
    lon = st.session_state.get("coord_lon", "")
    lat = st.session_state.get("coord_lat", "")
    selected_zoom = st.session_state.get("selected_zoom", 18)

    uploaded_image_path = st.session_state.get("uploaded_image_path", "")
    route_start = st.session_state.get("route_start", None)
    route_end = st.session_state.get("route_end", None)
    obstacles = st.session_state.get("obstacles_px", [])
    res_m_per_px = st.session_state.get("res_m_per_px", 0.55)
    obstacles_json = json.dumps(obstacles, ensure_ascii=False)

    popup_script = f"""
    <script>
    (function() {{
        let baseHref = window.location.href;

        try {{
            if (window.parent && window.parent.location && window.parent.location.href) {{
                baseHref = window.parent.location.href;
            }}
        }} catch (e) {{}}

        const url = new URL(baseHref);
        url.searchParams.set('page', '분석 결과');
        url.searchParams.set('equipment', {json.dumps(str(final_equipment))});
        url.searchParams.set('radius', {json.dumps(str(radius_km))});
        url.searchParams.set('place', {json.dumps(str(place_name))});
        url.searchParams.set('lon', {json.dumps(str(lon))});
        url.searchParams.set('lat', {json.dumps(str(lat))});
        url.searchParams.set('zoom', {json.dumps(str(selected_zoom))});

        // 추가: 시작점/도착점/장애물/해상도도 URL로 전달
        url.searchParams.set('img_path', {json.dumps(uploaded_image_path)});
        url.searchParams.set('start', {json.dumps(route_start)});
        url.searchParams.set('end', {json.dumps(route_end)});
        url.searchParams.set('obstacles', {json.dumps(obstacles_json)});
        url.searchParams.set('res', {json.dumps(res_m_per_px)});

        window.open(
            url.toString(),
            '_blank',
            'noopener,noreferrer,width=1200,height=800'
        );
    }})();
    </script>
    """
    html(popup_script, height=0, width=0)



# ============================
# 공통: 조건 요약 시 축척값 불러오는 함수
# ============================
def get_scale_label_from_radius(radius_km) -> str:
    """radius_km 값으로 전술지도 축척 라벨(1:25,000 …)을 역으로 찾아준다."""
    if radius_km is None:
        return "미입력"

    try:
        r = float(radius_km)
    except (TypeError, ValueError):
        return "미입력"

    # radius_km 값이 프리셋과 딱 맞게 떨어지지 않을 수도 있으니, 약간의 오차 허용
    for scale_label, cfg in TACTICAL_SCALE_PRESETS.items():
        if abs(cfg.get("radius_km", 0.0) - r) < 1e-6:
            return scale_label

    # 매칭되는 축척이 없으면 숫자 그대로 보여주기
    return f"{r} km"


# ============================
# 공통: 현재 입력 조건 요약 함수
# ============================
def render_condition_summary():
    """분석 결과 페이지에서 사용하는 입력 값 요약 블록."""
    radius_km = st.session_state.get("radius_km", None)
    place_name = st.session_state.get("place_name", "")
    lon = st.session_state.get("coord_lon", None)
    lat = st.session_state.get("coord_lat", None)
    show_current_location = st.session_state.get("show_current_location", False)
    final_equipment = st.session_state.get("final_equipment", "미지정 장비")

    if lon is not None and lat is not None:
        coords_text = f"{lon:.4f}, {lat:.4f}"
    else:
        coords_text = "미입력"

    scale_text = get_scale_label_from_radius(radius_km)

    st.markdown("##### ● 입력 값 요약")
    st.markdown(
        f"""
        - **전술지도 축척:** {scale_text}  
        - **지명:** {place_name or "미입력"}  
        - **좌표값:** {coords_text}  
        - **중앙 좌표 표시:** {"ON" if show_current_location else "OFF"}  
        - **분석 장비:** {final_equipment}  
        """
    )
