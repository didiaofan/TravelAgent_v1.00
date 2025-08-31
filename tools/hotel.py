from typing import Dict, Any, List
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



def ctrip_hotel_scraper(destination, checkin, checkout, rooms, adults, children, keyword=None, max_hotels=5):
    """
       从携程酒店列表页获取酒店数据

       参数：
           destination (str): 目的地
           checkin (str): 入住日期，格式 YYYY/MM/DD
           checkout (str): 退房日期，格式 YYYY/MM/DD
           rooms (int): 房间数
           adults (int): 成人数
           children (int): 儿童数
           keyword (str, 可选): 地址或酒店的进一步描述
           max_hotels (int): 需要抓取的酒店数量，默认 5

       返回：
           list[dict]: 酒店数据字典列表，包含：
               - 酒店名称
               - 评分
               - 房型
               - 价格
       """
    chrome_options = Options()
    chrome_options.debugger_address = "127.0.0.1:9222"
    driver = webdriver.Chrome(options=chrome_options)

    url = (
        f"https://hotels.ctrip.com/hotels/list?"
        f"city=1&provinceId=0&checkin={checkin}&checkout={checkout}"
        f"&optionId=1&optionType=City&directSearch=1"
        f"&optionName={destination}&display={destination}"
        f"&crn={rooms}&adult={adults}&children={children}&ages=0"
        f"&searchBoxArg=t&travelPurpose=0&domestic=1"
    )
    if keyword:
        url += f"&keyword={keyword}"

    driver.get(url)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.hotel-card")))

    hotels_data = []
    seen_hotels = set()

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_position = 0

    while len(hotels_data) < max_hotels:
        # 每次小幅度下滑，比如 500px
        scroll_position += 500
        driver.execute_script(f"window.scrollTo(0, {scroll_position});")
        time.sleep(1.5)  # 给页面时间加载

        hotels = driver.find_elements(By.CSS_SELECTOR, "div.hotel-card")
        for hotel in hotels:
            try:
                name = hotel.find_element(By.CSS_SELECTOR, ".hotelName").text
                if name in seen_hotels:
                    continue
                seen_hotels.add(name)

                score = hotel.find_element(By.CSS_SELECTOR, ".comment-score .score").text
                room_type = hotel.find_element(By.CSS_SELECTOR, ".room-name").text
                price = hotel.find_element(By.CSS_SELECTOR, ".room-price .sale").text

                hotels_data.append({
                    "酒店名称": name,
                    "评分": score,
                    "房型": room_type,
                    "价格": price
                })
                print(f"采集到第 {len(hotels_data)} 条：{name}")

                if len(hotels_data) >= max_hotels:
                    break
            except Exception:
                continue

        # 检查是否到底
        new_height = driver.execute_script("return document.body.scrollHeight")
        if scroll_position >= new_height:
            break

    return hotels_data
import json