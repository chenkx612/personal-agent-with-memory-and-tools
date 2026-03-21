"""Environment-related tools for time, location, and weather."""

import json
import urllib.request
import urllib.error
from datetime import datetime
from langchain_core.tools import tool

# Location cache
_location_cache = None

# Weekday names in Chinese
_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _get_location():
    """Get current location via IP geolocation. Cached for the session."""
    global _location_cache
    if _location_cache is not None:
        return _location_cache
    try:
        req = urllib.request.Request(
            "http://ipinfo.io/json",
            headers={"User-Agent": "personal-agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")
        loc = data.get("loc", "")  # "lat,lon"
        location_str = ", ".join(filter(None, [city, region, country]))
        lat, lon = None, None
        if loc and "," in loc:
            parts = loc.split(",")
            lat, lon = float(parts[0]), float(parts[1])
        _location_cache = {"location": location_str, "lat": lat, "lon": lon}
        return _location_cache
    except Exception:
        return None


def _get_weather(lat, lon):
    """Get current weather from Open-Meteo API (no API key required)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relative_humidity_2m"
            f"&forecast_days=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "personal-agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        wmo = cw.get("weathercode", 0)
        # Pick humidity at first hourly slot
        humidity = None
        hourly = data.get("hourly", {})
        rh_list = hourly.get("relative_humidity_2m", [])
        if rh_list:
            humidity = rh_list[0]
        # Map WMO weather code to description
        if wmo == 0:
            desc = "晴"
        elif wmo in (1, 2, 3):
            desc = "多云"
        elif wmo in range(45, 50):
            desc = "雾"
        elif wmo in range(51, 68):
            desc = "雨"
        elif wmo in range(71, 78):
            desc = "雪"
        elif wmo in range(80, 83):
            desc = "阵雨"
        elif wmo in range(95, 100):
            desc = "雷暴"
        else:
            desc = f"天气代码{wmo}"
        parts = []
        if temp is not None:
            parts.append(f"{temp}°C")
        parts.append(desc)
        if wind is not None:
            parts.append(f"风速 {wind} km/h")
        if humidity is not None:
            parts.append(f"湿度 {humidity}%")
        return ", ".join(parts)
    except Exception:
        return None


@tool
def get_environment_context():
    """Get current date, time, location and weather information.

    Returns structured environment context including:
    - Current date (with weekday) and time
    - User's approximate location based on IP
    - Current weather conditions
    """
    now = datetime.now()
    weekday = _WEEKDAY_ZH[now.weekday()]
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    lines = [
        f"日期: {date_str}（{weekday}）",
        f"时间: {time_str}",
    ]

    loc_info = _get_location()
    if loc_info and loc_info.get("location"):
        lines.append(f"位置: {loc_info['location']}")
        if loc_info.get("lat") is not None and loc_info.get("lon") is not None:
            weather = _get_weather(loc_info["lat"], loc_info["lon"])
            if weather:
                lines.append(f"天气: {weather}")

    return "\n".join(lines)
