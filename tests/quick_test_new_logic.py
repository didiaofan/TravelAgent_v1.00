#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试新的天气约束逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_classifier import WeatherClassifier, WeatherSuitability

def quick_test():
    """快速测试新逻辑的核心功能"""
    
    print("🧪 快速测试新的天气约束逻辑")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 测试数据
    test_pois = [
        {"name": "故宫博物院", "indoor": True, "suggested_duration_hours": 3.0},
        {"name": "八达岭长城", "indoor": False, "suggested_duration_hours": 4.0},
        {"name": "天坛", "indoor": False, "suggested_duration_hours": 2.5},
        {"name": "国家博物馆", "indoor": True, "suggested_duration_hours": 2.0},
    ]
    
    # 1. 测试极端天气检查
    print("\n1. 测试极端天气检查")
    extreme_weather = [
        {"fxDate": "2025-08-10", "textDay": "台风", "tempMax": "25", "tempMin": "18", "precip": "0.0"},
        {"fxDate": "2025-08-11", "textDay": "大风", "tempMax": "22", "tempMin": "15", "precip": "0.0"},
    ]
    weather_analysis = classifier.analyze_trip_weather(extreme_weather, ["2025-08-10", "2025-08-11"])
    is_blocked = classifier.check_extreme_weather_blocking(weather_analysis, 2)
    print(f"极端天气阻断: {'是' if is_blocked else '否'} (期望: 是)")
    
    # 2. 测试必去景点冲突
    print("\n2. 测试必去景点冲突") 
    rain_weather = [
        {"fxDate": "2025-08-10", "textDay": "大雨", "tempMax": "25", "tempMin": "18", "precip": "15.0"},
        {"fxDate": "2025-08-11", "textDay": "中雨", "tempMax": "22", "tempMin": "15", "precip": "8.0"},
    ]
    rain_analysis = classifier.analyze_trip_weather(rain_weather, ["2025-08-10", "2025-08-11"])
    must_visit = [{"name": "八达岭长城", "indoor": False}]
    has_conflict = classifier.check_must_visit_weather_conflict(rain_analysis, must_visit)
    print(f"必去景点冲突: {'是' if has_conflict else '否'} (期望: 是)")
    
    # 3. 测试景点筛选
    print("\n3. 测试景点筛选")
    filtered = classifier.filter_completely_inaccessible_pois(test_pois, rain_analysis)
    print(f"筛选结果: {len(test_pois)} → {len(filtered)} 个景点")
    print("保留景点:", [poi['name'] for poi in filtered])
    
    # 4. 测试饱满度检查
    print("\n4. 测试饱满度检查")
    # 稀少景点 - 应该不饱满
    sparse_pois = test_pois[:1]  # 只有故宫，3小时，2天24小时，差值21小时 > 10
    is_full, analysis = classifier.check_trip_fullness(sparse_pois, 12, 2)
    print(f"稀少行程饱满度: {'饱满' if is_full else '不饱满'} (差值: {analysis['time_difference']}h, 期望: 不饱满)")
    
    # 充实景点 - 应该饱满  
    is_full_rich, analysis_rich = classifier.check_trip_fullness(test_pois, 12, 2)
    print(f"充实行程饱满度: {'饱满' if is_full_rich else '不饱满'} (差值: {analysis_rich['time_difference']}h, 期望: 饱满)")
    
    print("\n✅ 核心功能测试完成!")

if __name__ == "__main__":
    quick_test()

