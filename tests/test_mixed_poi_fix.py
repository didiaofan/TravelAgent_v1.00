#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试混合型景点筛选修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_classifier import WeatherClassifier

def test_mixed_poi_filtering():
    """测试混合型景点在好天气下应该被保留"""
    
    print("🧪 测试混合型景点筛选修复")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 测试景点数据
    test_pois = [
        {"name": "北京环球影城", "indoor": "混合（室内外结合）", "suggested_duration_hours": 6.0},
        {"name": "故宫博物院", "indoor": True, "suggested_duration_hours": 3.0},
        {"name": "八达岭长城", "indoor": False, "suggested_duration_hours": 4.0},
        {"name": "王府井步行街", "indoor": "混合", "suggested_duration_hours": 2.0},
        {"name": "购物中心", "indoor": True, "suggested_duration_hours": 2.0},
    ]
    
    # 测试好天气（雷阵雨被归类为可户外出行）
    good_weather_data = [
        {"fxDate": "2025-08-17", "textDay": "雷阵雨", "tempMax": "30", "tempMin": "23", "precip": "0.0"},
        {"fxDate": "2025-08-18", "textDay": "雷阵雨", "tempMax": "31", "tempMin": "24", "precip": "0.0"},
    ]
    
    trip_dates = ["2025-08-17", "2025-08-18"]
    weather_analysis = classifier.analyze_trip_weather(good_weather_data, trip_dates)
    
    print(f"天气分析:")
    print(f"  适合户外天数: {weather_analysis['suitable_days']}")
    print(f"  室内天数: {weather_analysis['indoor_days']}")
    print(f"  极端天气天数: {weather_analysis['extreme_weather_days']}")
    
    # 进行景点筛选
    filtered_pois = classifier.filter_completely_inaccessible_pois(test_pois, weather_analysis)
    
    print(f"\n筛选结果:")
    print(f"原始景点数: {len(test_pois)}")
    print(f"筛选后景点数: {len(filtered_pois)}")
    
    print(f"\n保留的景点:")
    for poi in filtered_pois:
        indoor_status = poi.get("indoor", "未知")
        print(f"  ✓ {poi['name']} (室内状态: {indoor_status})")
    
    print(f"\n被移除的景点:")
    removed_pois = [poi for poi in test_pois if poi not in filtered_pois]
    for poi in removed_pois:
        indoor_status = poi.get("indoor", "未知")
        print(f"  ✗ {poi['name']} (室内状态: {indoor_status})")
    
    # 验证环球影城应该被保留
    universal_kept = any(poi['name'] == "北京环球影城" for poi in filtered_pois)
    print(f"\n🎯 关键验证:")
    print(f"北京环球影城是否保留: {'✅ 是' if universal_kept else '❌ 否'}")
    
    # 验证混合型景点都应该被保留
    mixed_pois = [poi for poi in test_pois if isinstance(poi.get("indoor"), str) and "混合" in poi.get("indoor")]
    mixed_kept = all(any(kept['name'] == mixed['name'] for kept in filtered_pois) for mixed in mixed_pois)
    print(f"所有混合型景点是否都保留: {'✅ 是' if mixed_kept else '❌ 否'}")
    
    return universal_kept and mixed_kept

if __name__ == "__main__":
    success = test_mixed_poi_filtering()
    print(f"\n{'🎉 测试通过!' if success else '❌ 测试失败!'}")

