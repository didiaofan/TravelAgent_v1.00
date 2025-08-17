#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证环球影城修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_classifier import WeatherClassifier

def verify_universal_studios_fix():
    """验证环球影城在雷阵雨天气下被保留"""
    
    classifier = WeatherClassifier()
    
    # 模拟您的场景
    test_pois = [
        {"name": "北京环球影城", "indoor": "混合（室内外结合）"},
        {"name": "故宫博物院", "indoor": True},
        {"name": "八达岭长城", "indoor": False},
    ]
    
    # 雷阵雨天气（现在被归类为可户外出行）
    weather_data = [
        {"fxDate": "2025-08-17", "textDay": "雷阵雨", "tempMax": "30", "tempMin": "23", "precip": "0.0"},
        {"fxDate": "2025-08-18", "textDay": "雷阵雨", "tempMax": "31", "tempMin": "24", "precip": "0.0"},
    ]
    
    weather_analysis = classifier.analyze_trip_weather(weather_data, ["2025-08-17", "2025-08-18"])
    filtered_pois = classifier.filter_completely_inaccessible_pois(test_pois, weather_analysis)
    
    print(f"天气: 雷阵雨 (适合户外天数: {weather_analysis['suitable_days']})")
    print(f"筛选结果: {len(test_pois)} → {len(filtered_pois)} 个景点")
    
    for poi in filtered_pois:
        print(f"✓ {poi['name']}")
    
    # 验证环球影城是否保留
    universal_kept = any(poi['name'] == "北京环球影城" for poi in filtered_pois)
    print(f"\n环球影城保留: {'✅ 是' if universal_kept else '❌ 否'}")
    
    return universal_kept

if __name__ == "__main__":
    verify_universal_studios_fix()

