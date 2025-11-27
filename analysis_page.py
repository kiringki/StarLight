# 결과 분석 페이지 (analysis_page3.py)

import json
import streamlit as st
import numpy as np
import pandas as pd
from typing import Callable
import os
from PIL import Image

from equipment_specs import EQUIPMENT_SPECS
from config import GOOGLE_MAPS_API_KEY
from map_api import get_google_satellite_image, estimate_zoom_from_radius

# ✅ 공통 코어 모듈 (사용자 장비 스펙)
from starlight_core import load_user_equipment_specs

# ✅ 경로 분석 모델 (routefinder)
from routefinder import analyze_routes_on_image


def render_analysis_page(render_condition_summary: Callable):
    """STARLIGHT - 분석 결과 페이지"""

    # ============================
    # 스타일: 밝은 군용 대시보드 테마
    # ============================
    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f3f4f6 !important;
        }

        .mil-card {
            border-radius: 10px;
            padding: 1rem 1.3rem;
            margin-bottom: 0.75rem;
            background: #ffffff;
            border: 1px solid #d1d5db;
        }

        .mil-card-mini {
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.4rem;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            font-size: 0.85rem;
        }

        .mil-label {
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .mil-value {
            font-size: 1.7rem;
            font-weight: 600;
            color: #111827;
        }

        .mil-chip {
            display: inline-block;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            font-size: 1.4rem;
            font-weight: 580;
        }
        .mil-chip-ok {
            background: #dcfce7;
            color: #166534;
            border: 1.5px solid #86efac;
        }
        .mil-chip-warn {
            background: #fef9c3;
            color: #854d0e;
            border: 1.5px solid #fde047;
        }
        .mil-chip-danger {
            background: #fee2e2;
            color: #991b1b;
            border: 1.5px solid #fca5a5;
        }

        thead tr th {
            background-color: #e5e7eb !important;
            color: #374151 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================
    # 쿼리파라미터 + 세션에서 입력값 가져오기
    # ============================
    qp = st.query_params
    img_path = qp.get("img_path", None)

    # 기본값은 세션에서, 있으면 쿼리값으로 override
    radius_km = st.session_state.get("radius_km", "미입력")
    final_equipment = st.session_state.get("final_equipment", "미지정 장비")
    place_name = st.session_state.get("place_name", "미지정 지역")

    if qp.get("radius"):
        try:
            radius_km = float(qp.get("radius"))
            st.session_state.radius_km = radius_km
        except ValueError:
            pass

    if qp.get("equipment"):
        final_equipment = qp.get("equipment")
        st.session_state.final_equipment = final_equipment

    if qp.get("place"):
        place_name = qp.get("place")
        st.session_state.place_name = place_name

    if qp.get("lon"):
        try:
            st.session_state.coord_lon = float(qp.get("lon"))
        except ValueError:
            pass

    if qp.get("lat"):
        try:
            st.session_state.coord_lat = float(qp.get("lat"))
        except ValueError:
            pass

    lat = st.session_state.get("coord_lat", 0.0)
    lon = st.session_state.get("coord_lon", 0.0)

    # ----------------------------
    # 좌표 파싱 유틸 (JSON ["x","y"] 형식 + "x,y" 문자열 둘 다 지원)
    # ----------------------------
    def parse_point(raw: str):
        if not raw:
            return None

        # 1) 먼저 JSON 형식 시도: 예) "[193, 851]"
        try:
            pt = json.loads(raw)
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                return (int(pt[0]), int(pt[1]))
        except Exception:
            pass

        # 2) "193,851" 같은 문자열 형식 파싱
        try:
            parts = raw.split(",")
            if len(parts) == 2:
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                return (x, y)
        except Exception:
            pass

        return None

    # ----------------------------
    # 경로 분석용 값들 (시작점 / 도착점 / 장애물 / 해상도)
    # ----------------------------
    route_start = None
    route_end = None
    obstacles_px = []

    start_raw = qp.get("start", None)
    end_raw = qp.get("end", None)
    obs_raw = qp.get("obstacles", None)

    # 시작점 / 도착점 파싱
    route_start = parse_point(start_raw) if start_raw and start_raw != "null" else None
    route_end = parse_point(end_raw) if end_raw and end_raw != "null" else None

    # 장애물 리스트 파싱
    if obs_raw and obs_raw != "null":
        try:
            # obstacles가 JSON 리스트 형식인 경우 (예: [[193,851],[200,300]])
            parsed = json.loads(obs_raw)
            if isinstance(parsed, list):
                obstacles_px = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        obstacles_px.append((int(item[0]), int(item[1])))
                    elif isinstance(item, str):
                        pt = parse_point(item)
                        if pt:
                            obstacles_px.append(pt)
        except Exception:
            # 혹시 "193,851;200,300" 이런 식으로 들어와도 안전하게 실패 처리
            obstacles_px = []
    else:
        obstacles_px = []

    # 해상도
    res_m_per_px = 0.55
    res_raw = qp.get("res", None)
    if res_raw is not None:
        try:
            res_m_per_px = float(res_raw)
        except ValueError:
            pass

    # ============================
    # 1) 분석용 이미지 가져오기
    #    - 1순위: 쿼리로 넘어온 업로드 이미지 경로 (이미지 분석 모드)
    #    - 2순위: 세션에 저장된 last_satellite_image
    #    - 3순위: 위도/경도로 Google Maps 재요청 (실시간 지도 모드)
    # ============================
    base_image = None
    model_error = None

    # 1) img_path 쿼리가 있으면, 업로드 이미지 모드라고 보고 파일에서 다시 로드
    if img_path:
        try:
            if os.path.exists(img_path):
                base_image = Image.open(img_path).convert("RGB")
                # 새 탭 세션에서도 재사용 가능하도록 다시 세션에 넣어줌
                st.session_state["last_satellite_image"] = base_image
        except Exception:
            base_image = None

    # 2) 세션에 이미지가 남아 있으면 그대로 사용
    if base_image is None:
        base_image = st.session_state.get("last_satellite_image", None)

    # 3) 그래도 없으면 → 실시간 지도 모드라고 보고 Google Maps에서 다시 가져오기
    if base_image is None:
        if not GOOGLE_MAPS_API_KEY:
            model_error = "Google Maps API 키가 설정되지 않았습니다."
        elif lat is None or lon is None:
            model_error = "메인 화면에서 좌표를 먼저 설정해야 합니다."
        else:
            try:
                radius_val = float(radius_km) if radius_km not in (None, "미입력") else 3.0
            except Exception:
                radius_val = 3.0

            zoom_q = qp.get("zoom")
            if zoom_q is not None:
                try:
                    zoom = int(float(zoom_q))
                except ValueError:
                    zoom = estimate_zoom_from_radius(radius_val)
            else:
                zoom = estimate_zoom_from_radius(radius_val)

            img = get_google_satellite_image(
                lat=lat,
                lng=lon,
                zoom=zoom,
                size="600x600",
                api_key=GOOGLE_MAPS_API_KEY,
            )
            if img is None:
                model_error = "위성 이미지를 불러오지 못했습니다. (API 키/좌표/쿼터를 확인하세요.)"
            else:
                base_image = img
                st.session_state["last_satellite_image"] = base_image

    # ============================
    # 2) 경로 분석 모델 호출
    # ============================
    model_result = None

    # 장비명 → 모델용 vehicle_name 매핑 (간단 매핑, 필요시 확장)
    vehicle_name = "K2 흑표"
    if "K9" in final_equipment:
        vehicle_name = "K9 자주포"
    elif "K808" in final_equipment:
        vehicle_name = "K808 장갑차"
    elif "소형" in final_equipment:
        vehicle_name = "소형전술차량"
    elif "두돈반" in final_equipment:
        vehicle_name = "두돈반"

    if model_error is None:
        # 이미지까지는 확보된 상태 → 시작/끝점 체크
        if not route_start or not route_end:
            model_error = "메인 화면에서 시작점과 도착점을 클릭해 설정해야 합니다."
        else:
            try:
                # routefinder.analyze_routes_on_image 호출
                model_result, msg = analyze_routes_on_image(
                    image=base_image,
                    vehicle_name=vehicle_name,
                    start_px=route_start,
                    end_px=route_end,
                    obstacles_px=obstacles_px,
                    res_m_per_px=res_m_per_px,
                )
                if model_result is None:
                    model_error = msg or "경로 분석에 실패했습니다."
            except Exception as e:
                model_error = f"경로 분석 중 오류 발생: {e}"

    # ============================
    # 3) 모델 결과를 기반으로 상단 핵심 지표 설정
    # ============================
    if model_result and "routes" in model_result:
        routes = model_result["routes"]
        available_routes = len(routes)

        # 가장 짧은 경로 길이 (m)
        if routes:
            min_dist = min(r.get("dist", 0.0) for r in routes)
        else:
            min_dist = 0.0

        # Mobility: 경로 존재 여부 + 거리 기준 단순 판단
        if available_routes == 0:
            mobility_label = "불가"
            mobility_chip_class = "mil-chip-danger"
        elif min_dist > 10000:  # 10km 이상이면 제한적
            mobility_label = "제한적 기동"
            mobility_chip_class = "mil-chip-warn"
        else:
            mobility_label = "가능"
            mobility_chip_class = "mil-chip-ok"

        # Risk: 장애물 개수 기반 단순 휴리스틱
        obstacle_count = len(model_result.get("obstacles", obstacles_px))
        if obstacle_count <= 1:
            risk_label = "안정"
            risk_chip_class = "mil-chip-ok"
        elif obstacle_count <= 4:
            risk_label = "주의"
            risk_chip_class = "mil-chip-warn"
        else:
            risk_label = "위험"
            risk_chip_class = "mil-chip-danger"

    else:
        routes = []
        available_routes = 0
        mobility_label = "불가"
        mobility_chip_class = "mil-chip-danger"
        risk_label = "위험"
        risk_chip_class = "mil-chip-danger"

    # ============================
    # 3-1) 경로 요약 테이블(route_data) 생성
    # ============================
    route_rows = []
    for r in routes:
        route_rows.append(
            {
                "경로 유형": r.get("type", "-"),
                "총 거리 (m)": round(r.get("dist", 0.0), 1),
            }
        )

    if route_rows:
        route_data = pd.DataFrame(route_rows)
    else:
        # 경로가 하나도 없을 때도 표는 비어 있지 않게 기본 한 줄 넣어줌
        route_data = pd.DataFrame(
            [{"경로 유형": "(경로 없음)", "총 거리 (m)": 0.0}]
        )

    # ============================
    # 상단: 제목 + 3개 주요 지표 카드 (한 줄 배치)
    # ============================
    col_title, col_cards = st.columns([1.4, 2.6])

    # --- 왼쪽: 제목 ---
    with col_title:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; justify-content:center; height:100%;">
                <div class="mil-label">STARLIGHT: 기동 가능성 평가 </div>
                <h2 style="margin:0.2rem 0; color:#1f2937;">분석 결과</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 오른쪽: 3개 주요 지표 카드 ---
    with col_cards:
        col_score, col_risk, col_route = st.columns([1.3, 1.1, 1.2])

        # --- Mobility 카드 ---
        with col_score:
            st.markdown('<div class="mil-card">', unsafe_allow_html=True)
            st.markdown('<div class="mil-label">Mobility</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="margin-top:0.4rem; font-size:0.95rem;">
                    <span class="mil-chip {mobility_chip_class}">{mobility_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Risk 카드 ---
        with col_risk:
            st.markdown('<div class="mil-card">', unsafe_allow_html=True)
            st.markdown('<div class="mil-label">Risk Level</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.6rem;">
                    <span class="mil-chip {risk_chip_class}">{risk_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Available Routes 카드 ---
        with col_route:
            st.markdown('<div class="mil-card">', unsafe_allow_html=True)
            st.markdown('<div class="mil-label">Available Routes</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mil-value">{available_routes} 개</div>',
                unsafe_allow_html=True,
            )
            st.caption("※ AI 모델 결과 기반 경로 수")
            st.markdown("</div>", unsafe_allow_html=True)


    # ============================
    # 중단: 경로 분석 테이블 + 지도 + 요약 (3단 레이아웃 복원)
    # ============================
    st.markdown("### ● 기동 경로 분석")

    # 레이아웃 비율: 지도(2) : 경로표(1) : 요약패널(1)
    col_map, col_routes, col_summary = st.columns([2, 1, 1])

    # ============================
    # 1. 좌측: 분석 대상 지역 위성 이미지 (최신 로직 적용)
    # ============================
    with col_map:

        DISPLAY_WIDTH = 750   # ✅ 여기 숫자로 화면 크기 조절

        if model_result and model_result.get("overlay_image") is not None:
            st.image(
                model_result["overlay_image"],
                width=DISPLAY_WIDTH   # ✅ use_container_width 제거
            )
        elif base_image is not None:
            st.image(
                base_image,
                caption="입력 위성 이미지 (모델 결과 없음)",
                width=DISPLAY_WIDTH   # ✅ use_container_width 제거
            )
        else:
            st.error("이미지를 불러올 수 없습니다.")


    # ============================
    # 2. 가운데: 경로 테이블 + 해석 가이드
    # ============================
    with col_routes:
        # route_data는 analysis_page3.py 상단에서 이미 계산되었다고 가정
        st.dataframe(route_data, use_container_width=True, height=240)

        with st.expander("경로 해석 가이드"):
            st.write(
                """
                - **총 거리 (m)**: 각 후보 경로의 예상 이동 거리  
                - **주요 제약 요소**: 지형·장애물 기반 자동 산출 예정  
                - **경로별 색상** : 최단(파랑), 최적(초록), 우회(주황)
                """
            )

    # ============================
    # 3. 우측: 미션 상태 + 입력/장비 요약
    # ============================
    with col_summary:
        # 1) 미션 상태 카드
        st.markdown('<div class="mil-card">', unsafe_allow_html=True)

        # 미션 상태 메시지 로직 (available_routes 변수 활용)
        if available_routes == 0:
            status_html = """
            <div class="mil-label">Mission Status</div>
            <div style="margin-top:0.4rem; font-size:0.9rem;">
                현재 조건 하에서는 <b>기동 불가</b>로 판단됩니다.
                <br/><br/>
                - 장애물 제거 또는 우회 경로 재설정 필요
            </div>
            """
        else:
            status_html = """
            <div class="mil-label">Mission Status</div>
            <div style="margin-top:0.4rem; font-size:0.9rem;">
                현재 분석 조건 하에서 <b>기동 가능</b>합니다.
                <br/><br/>
                - 최적/우회 경로 비교 검토 권장
            </div>
            """
        
        st.markdown(status_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2) 입력 값 요약 + 장비 제원 요약 (내부 2단 분리)
        st.markdown('<div class="mil-card">', unsafe_allow_html=True)
        col_sum, col_equi = st.columns(2)

        # --- 좌측: 입력 값 요약 ---
        with col_sum:
            render_condition_summary()

        # --- 우측: 장비 제원 요약 ---
        with col_equi:
            # 장비 제원 가져오기 로직
            final_eq = st.session_state.get("final_equipment", "미지정 장비")
            user_specs = load_user_equipment_specs()
            all_specs = {**EQUIPMENT_SPECS, **user_specs}
            specs = all_specs.get(final_eq, {})

            e_len = specs.get("length", 0.0)
            e_wid = specs.get("width", 0.0)
            e_wgt = specs.get("weight", 0.0)

            # 값이 없거나 0이면 '미입력' 처리
            txt_len = f"{e_len} m" if e_len else "미입력"
            txt_wid = f"{e_wid} m" if e_wid else "미입력"
            txt_wgt = f"{e_wgt} t" if e_wgt else "미입력"

            st.markdown("##### ● 장비 제원")
            st.markdown(
                f"""
                <div style="font-size:0.85rem; line-height:1.4;">
                - <b>명칭:</b> {final_eq}<br>
                - <b>길이:</b> {txt_len}<br>
                - <b>전폭:</b> {txt_wid}<br>
                - <b>중량:</b> {txt_wgt}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")