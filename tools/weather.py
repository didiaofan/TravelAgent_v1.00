import requests



def get_weather_7d(location_code, api_host,api_key):
    # 注意：免费用户使用开发环境API
    url = f"https://{api_host}/v7/weather/7d"

    params = {
        'location': location_code,
        'key': api_key  # 关键修改：API Key作为查询参数传递
    }

    response = requests.get(
        url,
        params=params,
        headers={'Accept-Encoding': 'gzip, deflate'}  # 保持压缩支持
    )
    return response
