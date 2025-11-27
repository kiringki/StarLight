# starlight_pages.py
# render_overview_page() : 실시간 지도 분석 페이지
# render_data_model_page() : 이미지 분석 페이지

import os
import numpy as np
import streamlit as st
from PIL import Image

from config import GOOGLE_MAPS_API_KEY
from equipment_specs import EQUIPMENT_SPECS
from map_api import (
    get_google_satellite_image,
    TACTICAL_SCALE_PRESETS,
    reverse_geocode,
)
from starlight_core import (
    load_user_equipment_specs,
    save_user_equipment_spec,
    open_analysis_popup,
)

# 이미지 위 좌표 클릭용 컴포넌트
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================
# 1) 실시간 지도 분석 페이지
# ============================
def render_overview_page():
    """실시간 지도 분석 페이지"""

    col_main, col_right = st.columns([4, 1])

    # 최초 진입 시 요구사항 패널 기본값
    if "show_requirement_panel" not in st.session_state:
        st.session_state.show_requirement_panel = False

    # ============================
    # 우측 설정 패널
    # ============================
    with col_right:
        st.markdown("###### ")
        st.markdown("#### 분석 설정")

        # ----------------------------
        # 장비 선택 + 스펙 자동/직접 입력
        # ----------------------------
        user_equipment_specs = load_user_equipment_specs()
        all_equipment_specs = {**EQUIPMENT_SPECS, **user_equipment_specs}

        equipment_options = list(all_equipment_specs.keys()) + ["사용자 입력"]

        equipment_type = st.selectbox(
            "◆ 분석 대상 장비",
            equipment_options,
            index=0,
            key="equipment_type",
        )

        custom_equipment = None

        if equipment_type == "사용자 입력":
            # 장비명 + 길이/폭/무게 직접 입력
            custom_equipment = st.text_input(
                "사용자 입력 장비명",
                placeholder="예: 상용 5톤 트럭, 신규 장비 코드명 등",
                key="custom_equipment",
            )

            col_len, col_wid, col_wgt = st.columns(3)
            with col_len:
                st.session_state.equip_length = st.number_input(
                    "장비길이(m)",
                    min_value=0.0,
                    max_value=50.0,
                    value=st.session_state.equip_length or 0.0,
                    step=0.1,
                    key="custom_equip_length",
                )
            with col_wid:
                st.session_state.equip_width = st.number_input(
                    "장비 폭(m)",
                    min_value=0.0,
                    max_value=20.0,
                    value=st.session_state.equip_width or 0.0,
                    step=0.1,
                    key="custom_equip_width",
                )
            with col_wgt:
                st.session_state.equip_weight = st.number_input(
                    "장비 중량(t)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.equip_weight or 0.0,
                    step=0.5,
                    key="custom_equip_weight",
                )

            col_z, col_x, col_c = st.columns([1, 3, 1])
            with col_x:
                # 장비 입력 상태일 때만 버튼 표시
                if custom_equipment:
                    if st.button("장비 제원 저장"):
                        save_user_equipment_spec(
                            custom_equipment,
                            st.session_state.equip_length,
                            st.session_state.equip_width,
                            st.session_state.equip_weight,
                        )
                        st.session_state["equip_saved"] = True
                else:
                    # 미리 정의된 장비 → 스펙 자동 로딩
                    specs = all_equipment_specs.get(equipment_type, {})
                    st.session_state.equip_length = specs.get("length", 0.0)
                    st.session_state.equip_width = specs.get("width", 0.0)
                    st.session_state.equip_weight = specs.get("weight", 0.0)

            if st.session_state.get("equip_saved"):
                st.success(
                    "장비 제원이 저장되었습니다.  \n다음부터 장비 목록에서 선택할 수 있습니다."
                )
                st.session_state["equip_saved"] = False

        else:
            # 미리 정의된 장비 → 스펙 자동 로딩
            specs = all_equipment_specs.get(equipment_type, {})
            st.session_state.equip_length = specs.get("length", 0.0)
            st.session_state.equip_width = specs.get("width", 0.0)
            st.session_state.equip_weight = specs.get("weight", 0.0)

        # 최종 장비명
        if equipment_type == "사용자 입력":
            final_equipment = (
                custom_equipment if custom_equipment else "미지정 사용자 입력 장비"
            )
        else:
            final_equipment = equipment_type

        st.session_state.final_equipment = final_equipment

        st.caption(f"현재 선택된 장비: **{final_equipment}**")

        # ----------------------------
        # 좌표 숫자 입력
        # ----------------------------
        st.markdown("◆ 좌표값 (경도, 위도)")
        col_lon, col_lat = st.columns(2)
        with col_lon:
            st.session_state.coord_lon = st.number_input(
                "경도 (longitude)",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.coord_lon,
                step=0.001,
                format="%.3f",
            )
        with col_lat:
            st.session_state.coord_lat = st.number_input(
                "위도 (latitude)",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.coord_lat,
                step=0.001,
                format="%.3f",
            )
        st.caption("소수점 셋째 자리까지 입력 가능  \n(예: 127.066, 38.241)")

        # ----------------------------
        # 전술지도 축척 선택
        # ----------------------------
        selected_scale = st.selectbox(
            "◆ 전술지도 축척 선택",
            list(TACTICAL_SCALE_PRESETS.keys()),
            index=1,  # 예: 1:50,000
        )

        preset = TACTICAL_SCALE_PRESETS[selected_scale]
        selected_zoom = preset["zoom"]
        selected_radius_km = preset["radius_km"]

        # session_state에 반영
        st.session_state["selected_zoom"] = selected_zoom
        st.session_state["radius_km"] = selected_radius_km

        st.caption(f"대략 반경: **{selected_radius_km} km**")

        # 중앙 좌표 표시 토글
        st.session_state.show_current_location = st.toggle(
            "중앙 좌표 표시",
            value=st.session_state.show_current_location,
            help="지도 상에 중앙 좌표 위치를 표시합니다.",
        )

        # ----------------------------
        # 지도 불러오기 버튼
        # ----------------------------
        col_1, col_2 = st.columns(2)
        with col_1:
            fetch_map = st.button("위성 지도  \n불러오기", key="btn_fetch_map")

        # ----------------------------
        # 요구사항 입력 패널 ON 버튼
        # ----------------------------
        with col_2:
            if st.button("요구사항  \n입력", key="btn_show_requirements"):
                st.session_state.show_requirement_panel = True

    # ============================
    # 좌측: 지도 표시 영역 + 클릭 UI
    # ============================
    with col_main:
        st.subheader("실시간 지도 분석")

        with st.container(border=True):
            st.markdown("### 위성 지도 뷰어")

            lon = st.session_state.get("coord_lon", 0.0)
            lat = st.session_state.get("coord_lat", 0.0)
            show_current_location = st.session_state.get("show_current_location", False)
            selected_zoom = st.session_state.get("selected_zoom", 16)

            api_key = GOOGLE_MAPS_API_KEY or ""

            # 지도 요청
            if "fetch_map" in locals() and fetch_map:
                if not api_key:
                    st.error("지도 API에 문제가 있습니다. GOOGLE_MAPS_API_KEY를 확인하세요.")
                else:
                    if lat == 0.0 and lon == 0.0:
                        st.warning("경도/위도를 먼저 입력해 주세요.")
                    else:
                        markers = None
                        if show_current_location:
                            markers = [f"color:red|size:mid|{lat},{lon}"]

                        img = get_google_satellite_image(
                            lat=lat,
                            lng=lon,
                            zoom=selected_zoom,
                            size="500x500",
                            api_key=api_key,
                            markers=markers,
                        )

                        if img:
                            st.session_state["last_satellite_image"] = img

                            # 위도/경도로 대략적인 지명 가져오기
                            place = reverse_geocode(lat, lon, api_key)
                            if place:
                                st.session_state.place_name = place
                            else:
                                st.warning("지명 정보를 자동으로 불러오지 못했습니다.")
                        else:
                            st.error(
                                "위성 이미지를 불러오지 못했습니다. API 키/쿼터/요청 파라미터를 확인하세요."
                            )

            place_name = st.session_state.get("place_name") or "미확인"

            st.markdown(
                f"""
                - 중앙 좌표: `{lat:.4f}, {lon:.4f}` / 지도 축척: `{selected_scale}`  
                - 지도 중앙 지명: `{place_name}`  
                """
            )

            # 마지막으로 불러온 이미지
            base_img = st.session_state.get("last_satellite_image", None)

            # ➊ 아직 지도 없을 때
            if base_img is None:
                st.info('우측 패널에서 **경도 / 위도 / 전술지도 축척**을 설정한 후 **"위성 지도 불러오기"** 버튼을 눌러 이미지를 가져옵니다.')


            # ➋ 지도는 뜨고, 아직 '요구사항 입력'을 안 눌렀을 때 → 그냥 이미지만 표시
            elif not st.session_state.get("show_requirement_panel", False):
                st.image(base_img, use_container_width=True)

            # ➌ 지도도 있고, 요구사항 입력 ON → 같은 자리에서 클릭 UI로 전환
            else:
                st.markdown("---")

                # 해상도 0.55 고정
                res = 0.55
                st.session_state["res_m_per_px"] = res

                # 현재 저장된 상태 불러오기
                route_start = st.session_state.get("route_start", None)
                route_end = st.session_state.get("route_end", None)
                obstacles = st.session_state.get("obstacles_px", [])

                # 직전 클릭 좌표 (없으면 None)
                last_click_xy = st.session_state.get("last_click_xy", None)

                # 🔹 상단 한 줄: 요구사항 설정 / 현재 설정된 값 / 결과 분석 버튼
                col_1, col_2, col_3 = st.columns([3, 2, 1])

                # ---------------------------
                # 왼쪽: 요구사항 설정 (모드 선택)
                # ---------------------------
                with col_1:
                    st.markdown("#### 경로분석 요구사항 설정")
                    mode = st.radio(
                        "클릭 모드 선택",
                        ["시작점 지정", "도착점 지정", "장애물 추가"],
                        horizontal=True,
                        key="route_click_mode",
                    )

                # ---------------------------
                # 가운데: 현재 설정된 값
                # ---------------------------
                with col_2:
                    st.markdown("#### 현재 설정된 값")
                    summary_container = st.container()

                # ---------------------------
                # 오른쪽: 결과 분석 버튼
                # ---------------------------
                with col_3:
                    st.markdown("#### ")
                    st.markdown(" ")  # 약간의 여백용
                    if st.button("결과 분석", key="btn_analyze_routes_main"):
                        if st.session_state.get("route_start") is None or st.session_state.get("route_end") is None:
                            st.error("시작점과 도착점을 모두 지정해 주세요.")
                        else:
                            open_analysis_popup()

                # ---------------------------
                # 아래쪽: 실제 지도 + 클릭 UI
                # ---------------------------
                st.caption("아래 이미지 위를 클릭하면, 선택된 모드에 따라 좌표가 저장됩니다.")

                # ✅ 분석용 원본 이미지 (모델에는 이 해상도 기준 좌표가 들어감)
                analysis_img = base_img

                # ✅ UI용 축소 이미지 (화면에 작게 보이게)
                display_size = 1000  # 여기 숫자 줄이면 화면에 더 작게 나옴
                display_img = analysis_img.resize((display_size, display_size))

                # ✅ 클릭은 축소된 display_img 기준으로 받음
                click = streamlit_image_coordinates(
                    display_img,
                    key="route_clicks",
                )

                if click is not None:
                    # ✅ 축소 이미지 좌표 → 원본 좌표로 변환
                    scale_x = analysis_img.width / display_img.width
                    scale_y = analysis_img.height / display_img.height

                    x = int(click["x"] * scale_x)
                    y = int(click["y"] * scale_y)
                    cur_xy = (x, y)

                    # 🔴 이전 클릭과 좌표가 같으면 (리런만 된 경우) 아무 것도 안 함
                    if cur_xy != last_click_xy:
                        mode_now = st.session_state.get("route_click_mode")
                        if mode_now == "시작점 지정":
                            st.session_state["route_start"] = cur_xy
                        elif mode_now == "도착점 지정":
                            st.session_state["route_end"] = cur_xy
                        else:  # 장애물 추가
                            obstacles = st.session_state.get("obstacles_px", []) + [cur_xy]
                            st.session_state["obstacles_px"] = obstacles

                        # 이번 클릭을 "마지막 클릭"으로 저장
                        st.session_state["last_click_xy"] = cur_xy

                # ✅ 클릭 처리 후, 최신 session_state 기준으로 요약 표시
                with summary_container:
                    st.write(f"• 시작점: `{st.session_state.get('route_start', None)}`")
                    st.write(f"• 도착점: `{st.session_state.get('route_end', None)}`")

                    obs_show = st.session_state.get("obstacles_px", [])
                    if obs_show:
                        st.write("• 장애물 좌표들:")
                        for idx, (ox, oy) in enumerate(obs_show, start=1):
                            st.write(f"  - #{idx}: ({ox}, {oy})")
                    else:
                        st.write("• 장애물: 없음")

                    if st.button("장애물 전체 초기화", key="btn_clear_obstacles"):
                        st.session_state["obstacles_px"] = []
                        st.success("장애물 좌표를 초기화했습니다.")



# ============================
# 2) 이미지 분석 페이지
# ============================
def render_data_model_page():
    """이미지 분석 페이지 (실시간 지도 분석과 동일한 UX)"""

    st.subheader("이미지 분석")

    # 기본 해상도(모델용) 고정
    st.session_state["res_m_per_px"] = 0.55

    # 전술지도 축척 기본값 (실시간 페이지와 동일하게 1:50,000 인덱스 1)
    scale_keys = list(TACTICAL_SCALE_PRESETS.keys())
    default_scale = scale_keys[1] if len(scale_keys) > 1 else scale_keys[0]
    if "selected_scale_image" not in st.session_state:
        st.session_state["selected_scale_image"] = default_scale

    # =========================
    # 1) 왼쪽: 이미지 업로드
    #    오른쪽: 장비 스펙 + 축척 (같은 줄)
    # =========================
    col_left, col_right = st.columns([2, 1])

    # ---------- 왼쪽: 이미지 ----------
    with col_left:
        st.markdown("#### 1) 위성 이미지 업로드")

        uploaded_file = st.file_uploader(
            "위성 이미지 또는 지도 데이터를 업로드하세요 (PNG, JPG 등)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="image_uploader",
        )

    if uploaded_file is not None:
        # ✅ 업로드된 파일을 PIL Image로 변환
        img = Image.open(uploaded_file).convert("RGB")

        # ✅ 모델 입력을 위해 1024x1024로 정규화
        analysis_size = 1024
        img_resized = img.resize((analysis_size, analysis_size))

        # ✅ 세션에 저장 → 현재 페이지 & 결과 페이지 공용 사용
        st.session_state["last_satellite_image"] = img_resized

        # ✅ 새 탭(결과 페이지)에서도 사용 가능하도록 "파일로 저장"
        os.makedirs("uploaded_images", exist_ok=True)
        save_path = os.path.join("uploaded_images", "image_analysis.png")
        img_resized.save(save_path)

        # ✅ 파일 경로를 세션에 저장 → open_analysis_popup()에서 URL로 전달됨
        st.session_state["uploaded_image_path"] = save_path


    else:
        base_img = st.session_state.get("last_satellite_image", None)

        if base_img is not None:
            st.image(
                base_img,
                caption="이전에 업로드한 이미지 (모델 입력용)",
                use_container_width=True,
            )
        else:
            st.info("먼저 분석에 사용할 위성 이미지를 업로드하세요.")


    # ---------- 오른쪽: 장비 스펙 + 전술지도 축척 ----------
    with col_right:
        st.markdown("#### 2) 장비 스펙")

        user_equipment_specs = load_user_equipment_specs()
        all_equipment_specs = {**EQUIPMENT_SPECS, **user_equipment_specs}
        equipment_options = list(all_equipment_specs.keys()) + ["사용자 입력"]

        # 한 줄에: 장비 선택 / 축척 선택
        col_eq, col_scale = st.columns([2, 1])

        # --- 장비 선택 / 입력 ---
        with col_eq:
            equipment_type_img = st.selectbox(
                "분석 대상 장비",
                equipment_options,
                key="equipment_type_image",
            )

        custom_equipment_name = None

        if equipment_type_img != "사용자 입력":
            specs = all_equipment_specs.get(equipment_type_img, {})
            st.session_state.equip_length = specs.get("length", 0.0)
            st.session_state.equip_width = specs.get("width", 0.0)
            st.session_state.equip_weight = specs.get("weight", 0.0)
            final_equipment = equipment_type_img
        else:
            # 사용자 입력 장비
            prev_name = st.session_state.get("final_equipment", "")
            default_name = "" if prev_name in all_equipment_specs else prev_name

            custom_equipment_name = st.text_input(
                "사용자 입력 장비명",
                value=default_name,
                placeholder="예: 상용 5톤 트럭, 신규 장비 코드명 등",
                key="custom_equipment_image",
            )

            col_len, col_wid, col_wgt = st.columns(3)
            with col_len:
                st.session_state.equip_length = st.number_input(
                    "길이 (m)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(st.session_state.get("equip_length", 0.0)),
                    step=0.1,
                    key="vehicle_length_image",
                )
            with col_wid:
                st.session_state.equip_width = st.number_input(
                    "폭 (m)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state.get("equip_width", 0.0)),
                    step=0.1,
                    key="vehicle_width_image",
                )
            with col_wgt:
                st.session_state.equip_weight = st.number_input(
                    "중량 (ton)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("equip_weight", 0.0)),
                    step=0.5,
                    key="vehicle_weight_image",
                )

            final_equipment = (
                custom_equipment_name if custom_equipment_name else "미지정 사용자 입력 장비"
            )

        # 장비 제원 요약
        st.session_state.final_equipment = final_equipment
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#4b5563; margin-top:0.5rem;">
                • 장비 길이: <b>{st.session_state.equip_length} m</b><br>
                • 장비 폭: <b>{st.session_state.equip_width} m</b><br>
                • 장비 중량: <b>{st.session_state.equip_weight} ton</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================
    # 3) 시작점 / 도착점 / 장애물 입력
    # =========================
    st.markdown("### 3) 시작점 / 도착점 / 장애물 입력")

    base_img = st.session_state.get("last_satellite_image", None)
    if base_img is None:
        st.warning("이미지를 업로드해야 좌표를 지정할 수 있습니다.")
        return

    # 분석용 원본 / UI용 축소 이미지 분리
    analysis_img = base_img
    display_img = analysis_img.resize((700, 700))

    col_l, col_r = st.columns([3, 1])

    # ---------- 좌측: 클릭 UI ----------
    with col_l:
        mode = st.radio(
            "클릭 모드",
            ["시작점 지정", "도착점 지정", "장애물 추가"],
            horizontal=True,
            key="route_click_mode_img",
        )

        click = streamlit_image_coordinates(
            display_img,
            key="image_clicks",
        )

        if click is not None:
            # 축소 이미지 → 원본 좌표로 변환
            scale_x = analysis_img.width / display_img.width
            scale_y = analysis_img.height / display_img.height

            x = int(click["x"] * scale_x)
            y = int(click["y"] * scale_y)
            cur_xy = (x, y)

            mode_now = st.session_state.get("route_click_mode_img")

            if mode_now == "시작점 지정":
                st.session_state["route_start"] = cur_xy
            elif mode_now == "도착점 지정":
                st.session_state["route_end"] = cur_xy
            else:
                obs = st.session_state.get("obstacles_px", [])
                st.session_state["obstacles_px"] = obs + [cur_xy]

        st.caption("※ 이미지를 클릭하면 선택된 모드에 따라 좌표가 저장됩니다.")

    # ---------- 우측: 현재 설정값 + 결과 분석 버튼 ----------
    with col_r:
        st.markdown("#### 현재 설정된 값")
        st.write("• 시작점:", st.session_state.get("route_start"))
        st.write("• 도착점:", st.session_state.get("route_end"))

        obs_show = st.session_state.get("obstacles_px", [])
        if obs_show:
            st.write("• 장애물:")
            for i, o in enumerate(obs_show, 1):
                st.write(f"  - #{i}: {o}")
        else:
            st.write("• 장애물: 없음")

        if st.button("장애물 전체 초기화", key="btn_clear_obstacles_image"):
            st.session_state["obstacles_px"] = []
            st.success("장애물 좌표를 초기화했습니다.")

        st.markdown("#### 결과 분석")

        if st.button("결과 분석 실행", key="btn_analyze_image"):
            if not st.session_state.get("route_start") or not st.session_state.get("route_end"):
                st.error("시작점과 도착점을 모두 지정해야 합니다.")
            else:
                # 실시간 지도 분석과 동일하게 팝업 호출
                open_analysis_popup()

    st.markdown("---")
