#Google 지도 API 관련

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from config import GOOGLE_MAPS_API_KEY



TACTICAL_SCALE_PRESETS = {
    "1:25,000 (상세 전술지도)": {
        "zoom": 19,
        "radius_km": 1.0,   # 대략 화면 중심 기준 1km 정도
    },
    "1:50,000 (작전지역 상세)": {
        "zoom": 18,
        "radius_km": 3.0,
    },
    "1:100,000 (광역 전술지도)": {
        "zoom": 15,
        "radius_km": 7.0,
    },
    "1:250,000 (광역 상황 파악)": {
        "zoom": 14,
        "radius_km": 20.0,
    },
    "1:500,000 (전구 상황 파악)": {
        "zoom": 13,
        "radius_km": 60.0,
    },
}


# 지도 가져오는 함수
def get_google_satellite_image(lat, lng, zoom=19, size="600x600", api_key=GOOGLE_MAPS_API_KEY, markers=None):
    """
    Google Maps Static API를 사용하여 위성 사진을 가져옵니다.

    :param lat: 위도
    :param lng: 경도
    :param zoom: 줌 레벨
    :param size: 이미지 크기 (ex. '600x600')
    :param api_key: Google Maps Static API 키
    :param markers: Static Maps markers 파라미터용 문자열 리스트 / 예: ["color:red|size:mid|37.5,127.0"]
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
   
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": "512x512",   # ✅ 이 값 × scale 이 실제 해상도
        "maptype": "satellite",
        "key": api_key,
        "scale": 2,          # ✅ 512 × 2 = 1024 × 1024
    }


    if markers:
        # 리스트를 받은 경우 '|'로 연결해서 하나의 문자열로
        # (Static Maps는 여러 markers 파라미터도 허용하지만, 여기선 단순화)
        params["markers"] = "|".join(markers)

   
    response = requests.get(base_url, params=params)
   
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def estimate_zoom_from_radius(radius_km: float) -> int:
    """
    반경(km)에 따라 대략적인 줌 레벨을 추정.
    값은 Static Map용이라 대략적인 체감 기준으로 설정.
    """
    if radius_km <= 1:
        return 18
    elif radius_km <= 2:
        return 17
    elif radius_km <= 4:
        return 16
    elif radius_km <= 8:
        return 15
    elif radius_km <= 15:
        return 14
    elif radius_km <= 30:
        return 13
    elif radius_km <= 60:
        return 12
    elif radius_km <= 120:
        return 11
    else:
        return 10


# 지명 불러오는 함수
def reverse_geocode(lat: float, lng: float, api_key: str, language: str = "ko") -> str | None:
    """
    위도/경도로 대략적인 지명을 가져오는 함수 (Google Geocoding API 사용)

    반환값 예시: '대한민국 서울특별시 영등포구 여의도동 ...'
    """
    if not api_key:
        return None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key,
        "language": language,  # 'ko'로 하면 한글 주소
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        # 가장 첫 번째 결과의 전체 주소 사용
        return results[0].get("formatted_address")

    except Exception:
        return None

