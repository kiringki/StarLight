# routefinder.py
# - Colab 의존성 제거
# - CommandCenterSystem (도로/건물 세그먼트 + 경로 탐색)
# - analyze_routes_on_image / create_result_figure 제공

import os
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from PIL import Image

from scipy.ndimage import distance_transform_edt
from skimage.morphology import closing, square, dilation, disk
from skimage.graph import MCP_Geometric

import matplotlib.pyplot as plt

# (선택) 한글 폰트 설정 - NanumGothic 설치되어 있을 때만 의미 있음
plt.rcParams["font.family"] = "NanumGothic"


class CommandCenterSystem:
    """
    도로 / 건물 세그멘테이션 + 전술 경로 탐색을 담당하는 클래스.
    """

    def __init__(self, road_pth: str, bldg_pth: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.road_model = self._load_model(road_pth)
        self.bldg_model = self._load_model(bldg_pth)

        # 기존 코드에서 사용하던 차량 폭 정보 (v_name 사용할 경우를 위해 남겨둠)
        self.vehicles = {
            "K2 흑표": 3.6,
            "K9 자주포": 3.4,
            "K808 장갑차": 2.7,
            "소형전술차량": 2.2,
            "두돈반": 2.4,
        }

    def _load_model(self, path: str):
        """
        저장된 .pth 세그멘테이션 모델 로드
        """
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        ).to(self.device)

        if path and os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            return model

        # 파일이 없으면 None
        return None

    def analyze_terrain(self, img_arr: np.ndarray):
        """
        위성 RGB 이미지 (H, W, 3) -> 도로/건물 바이너리 마스크
        r_bin: 도로, b_bin: 건물
        """
        h, w = img_arr.shape[:2]
        inp = cv2.resize(img_arr, (1024, 1024))
        tensor = torch.tensor(
            inp.transpose(2, 0, 1) / 255.0, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        if self.road_model is None or self.bldg_model is None:
            raise RuntimeError("도로/건물 세그멘테이션 모델이 로드되지 않았습니다.")

        with torch.no_grad():
            r_mask = self.road_model(tensor).squeeze().cpu().numpy()
            b_mask = self.bldg_model(tensor).squeeze().cpu().numpy()

        # 원래 해상도로 리사이즈
        r_mask = cv2.resize(r_mask, (w, h))
        b_mask = cv2.resize(b_mask, (w, h))

        # 도로는 closing + threshold
        r_bin = closing((r_mask > 0.4).astype(np.uint8), square(3))
        # 건물은 threshold만
        b_bin = (b_mask > 0.4).astype(np.uint8)

        return r_bin, b_bin

    def calculate_tactical_routes(
        self,
        r_mask: np.ndarray,
        b_mask: np.ndarray,
        user_obstacles: list[tuple[int, int]] | None,
        start: tuple[int, int],
        end: tuple[int, int],
        res: float = 0.55,
        v_name: str | None = None,
        vehicle_width: float | None = None,
    ):
        """
        - r_mask: 도로 바이너리 맵 (1: 도로)
        - b_mask: 건물 바이너리 맵 (1: 건물)
        - user_obstacles: (x, y) 리스트
        - start, end: (x, y) 픽셀 좌표
        - res: 해상도 (m/px)
        - v_name: 차량 이름
        - vehicle_width: 차량 폭(m)을 직접 지정할 때 사용 (우선순위 더 높음)
        """
        user_obstacles = user_obstacles or []

        # ✅ 1) 시작/도착점과 너무 가까운 장애물은 자동 제거 (안전 여유)
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        min_r = 25
        min_r2 = min_r * min_r

        cleaned_obstacles: list[tuple[int, int]] = []
        for (ox, oy) in user_obstacles:
            ox, oy = int(ox), int(oy)
            if (ox, oy) == (sx, sy) or (ox, oy) == (ex, ey):
                continue
            if (ox - sx) ** 2 + (oy - sy) ** 2 < min_r2:
                continue
            if (ox - ex) ** 2 + (oy - ey) ** 2 < min_r2:
                continue
            cleaned_obstacles.append((ox, oy))

        user_obstacles = cleaned_obstacles

        # ============================
        # 2) 통행 가능 영역(passable) 정의
        #    → "오렌지색 도로 세그멘테이션을 넉넉하게 확장"
        # ============================
        # 기본 도로 영역
        passable = r_mask.astype(np.uint8)
        # 도로를 조금 두껍게 만들어서 끊어진 곳을 연결
        passable = dilation(passable, disk(2))

        # 도로/비도로의 경계 정보를 위한 거리맵 (안전 경로 가중치용)
        dist_map = distance_transform_edt(passable == 0)

        # 동적 장애물 맵
        obstacle_map = np.zeros_like(passable, dtype=np.uint8)
        for obs in user_obstacles:
            x, y = int(obs[0]), int(obs[1])
            cv2.circle(obstacle_map, (x, y), 20, 1, -1)

        # ============================
        # 3) 시작/도착점 스냅
        #    → 세그멘테이션된 도로 근처면 도로 위로 붙여주고,
        #      그래도 못 찾으면 사용자가 찍은 좌표 그대로 사용
        # ============================
        snapped_start = self._snap_to_road(start, passable)
        snapped_end = self._snap_to_road(end, passable)

        if snapped_start is not None:
            start = snapped_start
        if snapped_end is not None:
            end = snapped_end

        # 화면 밖만 막아주고, 건물 위/장애물 위라도 "비용"으로 처리 (완전 차단 X)
        h, w = passable.shape

        def _out_of_bounds(pt):
            x, y = int(pt[0]), int(pt[1])
            return x < 0 or x >= w or y < 0 or y >= h

        if _out_of_bounds(start) or _out_of_bounds(end):
            # 이 경우만 진짜 에러 처리
            return None, "출발지 또는 도착지가 지도 범위를 벗어났습니다."

        # ============================
        # 4) 비용 맵 구성
        # ============================
        BASE_COST = 1.0          # 도로 위 기본 비용
        OFFROAD_COST = 10.0      # 도로 밖 (비도로) 비용
        BUILDING_COST = 20.0     # 건물 위 비용 (갈 수는 있으나 최대한 피함)

        cost_short = np.full_like(passable, OFFROAD_COST, dtype=float)
        cost_opt = np.full_like(passable, OFFROAD_COST, dtype=float)

        road_region = (passable == 1)

        # ① 최단 경로: 도로는 1, 나머지는 10, 건물은 20
        cost_short[road_region] = BASE_COST
        cost_short[b_mask == 1] = BUILDING_COST

        # ② 안전 경로: 도로 중심부일수록 비용 낮게
        safe_weight = distance_transform_edt(passable == 0)
        inside = road_region

        # 도로 내부: 1 + (5 / (중심부까지 거리+1)) → 도로 안에서도 가운데일수록 조금 더 저렴
        cost_opt[inside] = 1.0 + (5.0 / (safe_weight[inside] + 1.0))
        # 도로 밖: 오프로드
        cost_opt[~inside] = OFFROAD_COST
        # 건물 위: 더 비싸게
        cost_opt[b_mask == 1] = BUILDING_COST

        # ③ 동적 장애물: 완전 차단
        cost_short[obstacle_map == 1] = np.inf
        cost_opt[obstacle_map == 1] = np.inf

        # ============================
        # 5) MCP로 경로 탐색
        # ============================
        def solve_mcp(cost_grid):
            mcp = MCP_Geometric(cost_grid)
            try:
                mcp.find_costs(
                    starts=[(int(start[1]), int(start[0]))],
                    ends=[(int(end[1]), int(end[0]))],
                )
                path = mcp.traceback(end=(int(end[1]), int(end[0])))
                if path is None:
                    return None
                return [(int(p[1]), int(p[0])) for p in path]
            except Exception:
                return None

        p1 = solve_mcp(cost_short)  # 최단
        p2 = solve_mcp(cost_opt)    # 안전

        # ③ 우회 경로 (최적 경로 주변에 페널티)
        p3 = None
        if p2:
            cost_detour = cost_opt.copy()
            self._apply_penalty(cost_detour, p2)
            p3 = solve_mcp(cost_detour)

        results: list[dict] = []
        if p1:
            results.append(
                {
                    "type": "최단 경로",
                    "path": p1,
                    "color": "blue",
                    "style": ":",
                    "dist": self.calc_dist(p1, res),
                }
            )
        if p2:
            results.append(
                {
                    "type": "최적 경로",
                    "path": p2,
                    "color": "#00FF00",
                    "style": "-",
                    "dist": self.calc_dist(p2, res),
                }
            )
        if p3:
            results.append(
                {
                    "type": "우회 경로",
                    "path": p3,
                    "color": "orange",
                    "style": "--",
                    "dist": self.calc_dist(p3, res),
                }
            )

        # 여기서는 result를 None으로 절대 돌려보내지 않고,
        # 경로가 없으면 routes=[] 인 dict로 넘긴다.
        msg = "성공" if results else "경로를 찾지 못했습니다."
        return {
            "routes": results,
            "start": start,
            "end": end,
            "obstacles": user_obstacles,
        }, msg

    def _apply_penalty(self, cost_map, path):
        path_mask = np.zeros_like(cost_map, dtype=np.uint8)
        for px, py in path:
            if 0 <= py < path_mask.shape[0] and 0 <= px < path_mask.shape[1]:
                path_mask[py, px] = 1
        penalty_zone = dilation(path_mask, disk(15))
        cost_map[penalty_zone == 1] *= 10.0

    @staticmethod
    def calc_dist(path, res):
        if not path or len(path) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(path) - 1):
            dist += np.sqrt(
                (path[i][0] - path[i + 1][0]) ** 2
                + (path[i][1] - path[i + 1][1]) ** 2
            )
        return dist * res

    @staticmethod
    def _snap_to_road(pt, mask, search_range=50):
        x0, y0 = int(pt[0]), int(pt[1])
        h, w = mask.shape

        if 0 <= x0 < w and 0 <= y0 < h and mask[y0, x0] == 1:
            return (x0, y0)

        y_min = max(0, y0 - search_range)
        y_max = min(h, y0 + search_range)
        x_min = max(0, x0 - search_range)
        x_max = min(w, x0 + search_range)

        sub_mask = mask[y_min:y_max, x_min:x_max]
        y_idxs, x_idxs = np.where(sub_mask == 1)
        if len(y_idxs) == 0:
            return None

        y_idxs = y_idxs + y_min
        x_idxs = x_idxs + x_min
        dists = (y_idxs - y0) ** 2 + (x_idxs - x0) ** 2
        idx = np.argmin(dists)
        return (int(x_idxs[idx]), int(y_idxs[idx]))


# ============================
# 전역 CommandCenterSystem (한 번만 로드)
# ============================

# 🔴 모델 절대경로
ROAD_PTH = r"C:\Users\User\Desktop\Starlight\models\best_road_model.pth"
BLDG_PTH = r"C:\Users\User\Desktop\Starlight\models\best_building_model.pth"

# 전역 객체 (캐시)
_CMD_SYSTEM: CommandCenterSystem | None = None


def get_system():
    global _CMD_SYSTEM
    if _CMD_SYSTEM is None:
        print("[DEBUG] Load CommandCenterSystem...")
        _CMD_SYSTEM = CommandCenterSystem(
            road_pth=ROAD_PTH,
            bldg_pth=BLDG_PTH,
        )
    return _CMD_SYSTEM


# Streamlit에서 사용할 시스템
system = get_system()


def _get_default_system() -> CommandCenterSystem:
    """
    routefinder 내부에서 전역 CommandCenterSystem을 한 번만 만들어서 재사용.
    """
    global _CMD_SYSTEM
    if _CMD_SYSTEM is None:
        _CMD_SYSTEM = CommandCenterSystem(ROAD_PTH, BLDG_PTH)
    return _CMD_SYSTEM


def analyze_routes_on_image(
    image,
    vehicle_name: str,
    start_px: tuple[int, int],
    end_px: tuple[int, int],
    obstacles_px: list[tuple[int, int]] | None = None,
    res_m_per_px: float = 0.55,
):
    """
    Streamlit 분석 페이지에서 바로 호출할 수 있는 래퍼 함수.
    """
    # 전역 시스템 가져오기
    system = _get_default_system()

    if obstacles_px is None:
        obstacles_px = []

    # 이미지 → numpy RGB
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = np.asarray(image)

    # 1) 도로/건물 세그멘테이션
    try:
        r_mask, b_mask = system.analyze_terrain(img)
    except RuntimeError as e:
        # 모델 자체를 못 불렀을 때는 진짜로 실패
        return None, str(e)

    # 2) 차량 폭 결정 (현재는 내부에서 직접 사용 X, 필요시 확장용)
    vehicle_width_m = system.vehicles.get(vehicle_name, 3.0)

    # 3) 경로 계산
    result, msg = system.calculate_tactical_routes(
        r_mask=r_mask,
        b_mask=b_mask,
        user_obstacles=obstacles_px,
        start=start_px,
        end=end_px,
        res=res_m_per_px,
        v_name=vehicle_name,
        vehicle_width=vehicle_width_m,
    )

    # ============================
    # 여기부터: "경로가 없어도" 세그멘테이션 + 시작/끝/장애물은 그려서 보여주기
    # ============================
    vis = img.copy()

    # 도로 부분을 살짝 오렌지색으로 오버레이
    overlay = vis.copy()
    overlay[r_mask == 1] = [255, 165, 0]  # RGB 오렌지
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 시작/끝점 표시 (입력 좌표 기준)
    sx, sy = int(start_px[0]), int(start_px[1])
    ex, ey = int(end_px[0]), int(end_px[1])
    cv2.circle(vis, (sx, sy), 8, (255, 255, 0), -1)  # 시작(노랑)
    cv2.circle(vis, (ex, ey), 8, (0, 0, 255), -1)    # 끝(파랑)

    # result가 있는 경우에는 실제 스냅된 좌표와 경로로 한 번 더 덮어그리기
    routes = []
    start_for_result = start_px
    end_for_result = end_px
    obstacles_for_result = obstacles_px

    if result is not None:
        routes = result.get("routes", [])
        start_for_result = result.get("start", start_px)
        end_for_result = result.get("end", end_px)
        obstacles_for_result = result.get("obstacles", obstacles_px)

        # 스냅된 시작/끝점으로 다시 그려주고
        sx, sy = int(start_for_result[0]), int(start_for_result[1])
        ex, ey = int(end_for_result[0]), int(end_for_result[1])
        cv2.circle(vis, (sx, sy), 8, (255, 255, 0), -1)
        cv2.circle(vis, (ex, ey), 8, (0, 0, 255), -1)

        for (ox, oy) in obstacles_for_result:
            ox, oy = int(ox), int(oy)
            cv2.drawMarker(
                vis,
                (ox, oy),
                color=(255, 0, 0),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=22,      # ← 여기 숫자 키우면 더 크게 보임
                thickness=3,        # ← 굵기
            )


        # 경로별 색깔 설정
        def _route_color(route_type: str):
            if "최단" in route_type:
                return (0, 0, 255)      # 파랑
            if "최적" in route_type:
                return (0, 255, 0)      # 초록
            if "우회" in route_type:
                return (255, 165, 0)    # 오렌지
            return (255, 255, 255)

        for r in routes:
            path = r.get("path") or []
            if len(path) < 2:
                continue
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            color = _route_color(r.get("type", ""))
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=9)


    overlay_image = Image.fromarray(vis)


    model_result = {
        "routes": routes,                     # 경로가 없으면 빈 리스트
        "start": start_for_result,           # 스냅되었으면 스냅 좌표, 아니면 입력 좌표
        "end": end_for_result,
        "obstacles": obstacles_for_result,
        "r_mask": r_mask,
        "b_mask": b_mask,
        "overlay_image": overlay_image,
    }

    return model_result, msg
