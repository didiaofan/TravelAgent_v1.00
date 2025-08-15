
import requests

def geocode_address(api_key, address):
    """
    将地址转换为经纬度坐标
    """
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": api_key,
        "address": address
    }
    res = requests.get(url, params=params).json()
    if res.get("status") == "1" and res.get("geocodes"):
        location = res["geocodes"][0]["location"]  # "lon,lat"
        return tuple(map(float, location.split(",")))
    return None


def get_route_info(api_key, origin_addr, destination_addr):
    """
    获取两个地址之间的出行信息（公共交通 & 出租车）

    功能:
        先将出发地和目的地地址解析为经纬度，再调用高德路线 API
        获取公共交通最短时间/费用 & 出租车最短时间/费用。

    参数:
        api_key (str): 高德 API Key
        origin_addr (str): 出发地地址
        destination_addr (str): 目的地地址

    返回:
        dict: {
            "出发地": str,
            "目的地": str,
            "公共交通最短时间": float (分钟),
            "公共交通费用": str,
            "出租车最短时间": float (分钟),
            "出租车费用": str
        }
    """
    # 1. 地址转经纬度
    origin_coords = geocode_address(api_key, origin_addr)
    dest_coords = geocode_address(api_key, destination_addr)

    if not origin_coords or not dest_coords:
        raise ValueError("地理编码失败，请检查输入地址")

    origin = f"{origin_coords[0]},{origin_coords[1]}"
    destination = f"{dest_coords[0]},{dest_coords[1]}"

    # 2. 公共交通方案
    bus_url = "https://restapi.amap.com/v3/direction/transit/integrated"
    bus_params = {
        "key": api_key,
        "origin": origin,
        "destination": destination,
        "city": "北京",
        "cityd": "北京",
        "strategy": 0  # 0 = 最快捷
    }
    bus_res = requests.get(bus_url, params=bus_params).json()

    bus_time = None
    bus_cost = None
    if bus_res.get("status") == "1" and bus_res.get("route", {}).get("transits"):
        fastest_transit = min(bus_res["route"]["transits"], key=lambda x: float(x["duration"]))
        bus_time = round(float(fastest_transit["duration"]) / 60, 1)
        bus_cost = fastest_transit.get("cost", "0") + "元"

    # 3. 出租车方案
    taxi_url = "https://restapi.amap.com/v3/direction/driving"
    taxi_params = {
        "key": api_key,
        "origin": origin,
        "destination": destination
    }
    taxi_res = requests.get(taxi_url, params=taxi_params).json()

    taxi_time = None
    taxi_cost = None
    if taxi_res.get("status") == "1" and taxi_res.get("route", {}).get("paths"):
        fastest_taxi = min(taxi_res["route"]["paths"], key=lambda x: float(x["duration"]))
        taxi_time = round(float(fastest_taxi["duration"]) / 60, 1)
        taxi_cost = taxi_res["route"].get("taxi_cost", "0") + "元"

    return {
        "出发地": origin_addr,
        "目的地": destination_addr,
        "公共交通最短时间": bus_time,
        "公共交通费用": bus_cost,
        "出租车最短时间": taxi_time,
        "出租车费用": taxi_cost
    }
