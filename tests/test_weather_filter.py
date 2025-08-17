#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 weather_filter 节点 - 新的天气约束流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_classifier import WeatherClassifier, WeatherSuitability, format_weather_analysis

def test_weather_classifier():
    """测试天气分类器"""
    
    print("🌤️ 测试天气分类器")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 测试各种天气描述
    test_weather_cases = [
        # 可户外出行的天气
        ("晴", WeatherSuitability.OUTDOOR_SUITABLE),
        ("多云", WeatherSuitability.OUTDOOR_SUITABLE),
        ("阴", WeatherSuitability.OUTDOOR_SUITABLE),
        ("薄雾", WeatherSuitability.OUTDOOR_SUITABLE),
        
        # 可市内出行的天气  
        ("小雨", WeatherSuitability.INDOOR_SUITABLE),
        ("中雨", WeatherSuitability.INDOOR_SUITABLE),
        ("雷阵雨", WeatherSuitability.INDOOR_SUITABLE),
        ("小雪", WeatherSuitability.INDOOR_SUITABLE),
        
        # 不建议出行的天气
        ("大风", WeatherSuitability.NOT_RECOMMENDED),
        ("沙尘暴", WeatherSuitability.NOT_RECOMMENDED),
        ("台风", WeatherSuitability.NOT_RECOMMENDED),
        ("暴雨", WeatherSuitability.INDOOR_SUITABLE),  # 暴雨应该是室内
    ]
    
    print("🔍 天气分类测试:")
    for weather_text, expected in test_weather_cases:
        result = classifier.classify_weather(weather_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} {weather_text} → {result.value} (预期: {expected.value})")
    
    return classifier

def test_poi_filtering():
    """测试景点筛选功能"""
    
    print("\n🏛️ 测试景点筛选功能")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 模拟候选景点数据
    test_pois = [
        {"name": "故宫博物院", "indoor": True, "tags": ["历史", "博物馆"]},
        {"name": "北京环球影城", "indoor": "混合（室内外结合）", "tags": ["主题乐园"]},
        {"name": "八达岭长城", "indoor": False, "tags": ["古迹", "徒步"]},
        {"name": "国家博物馆", "indoor": True, "tags": ["博物馆", "文化"]},
        {"name": "天坛", "indoor": False, "tags": ["古建筑", "公园"]},
        {"name": "颐和园", "indoor": False, "tags": ["皇家园林", "湖泊"]},
        {"name": "王府井步行街", "indoor": "混合", "tags": ["购物", "商业街"]},
        {"name": "中国科技馆", "indoor": True, "tags": ["科技馆", "教育"]},
    ]
    
    weather_scenarios = [
        (WeatherSuitability.OUTDOOR_SUITABLE, "晴天"),
        (WeatherSuitability.INDOOR_SUITABLE, "降水天气"),
        (WeatherSuitability.NOT_RECOMMENDED, "极端天气")
    ]
    
    for suitability, scenario_name in weather_scenarios:
        print(f"\n📊 {scenario_name}下的景点筛选:")
        filtered_pois = classifier.get_suitable_pois(test_pois, suitability)
        
        print(f"原始景点数: {len(test_pois)}")
        print(f"筛选后景点数: {len(filtered_pois)}")
        
        if filtered_pois:
            print("保留的景点:")
            for poi in filtered_pois:
                indoor_status = poi.get("indoor", "未知")
                print(f"  ✓ {poi['name']} (室内状态: {indoor_status})")
        else:
            print("  (无推荐景点)")

def test_trip_weather_analysis():
    """测试行程天气分析"""
    
    print("\n📅 测试行程天气分析")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 模拟7天天气数据
    mock_weather_data = [
        {"fxDate": "2025-08-10", "textDay": "暴雨", "tempMax": "30", "tempMin": "20", "precip": "0.0"},
        {"fxDate": "2025-08-11", "textDay": "台风", "tempMax": "28", "tempMin": "19", "precip": "0.0"},
        {"fxDate": "2025-08-12", "textDay": "小雨", "tempMax": "25", "tempMin": "18", "precip": "5.2"},
        {"fxDate": "2025-08-13", "textDay": "中雨", "tempMax": "23", "tempMin": "17", "precip": "12.5"},
        {"fxDate": "2025-08-14", "textDay": "晴", "tempMax": "29", "tempMin": "21", "precip": "0.0"},
        {"fxDate": "2025-08-15", "textDay": "阴", "tempMax": "26", "tempMin": "19", "precip": "0.0"},
        {"fxDate": "2025-08-16", "textDay": "雷阵雨", "tempMax": "24", "tempMin": "18", "precip": "8.3"},
    ]
    
    # 测试2天行程
    trip_dates = ["2025-08-10", "2025-08-11"]
    analysis = classifier.analyze_trip_weather(mock_weather_data, trip_dates)
    
    print("📊 2天行程天气分析结果:")
    weather_report = format_weather_analysis(analysis)
    print(weather_report)
    
    # 测试5天行程（包含降水）
    trip_dates_5d = ["2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14"]
    analysis_5d = classifier.analyze_trip_weather(mock_weather_data, trip_dates_5d)
    
    print("\n📊 5天行程天气分析结果:")
    weather_report_5d = format_weather_analysis(analysis_5d)
    print(weather_report_5d)

def test_weather_filter_node_simulation():
    """模拟测试weather_filter节点（不调用真实API）"""
    
    print("\n🔧 模拟测试 weather_filter 节点")
    print("=" * 50)
    
    # 模拟state数据
    test_state = {
        "candidate_pois": [
            {"name": "故宫博物院", "indoor": True, "computed_score": 1.98},
            {"name": "北京环球影城", "indoor": "混合（室内外结合）", "computed_score": 1.99},
            {"name": "八达岭长城", "indoor": False, "computed_score": 0.95},
            {"name": "天坛", "indoor": False, "computed_score": 0.85},
            {"name": "国家博物馆", "indoor": True, "computed_score": 0.86},
        ],
        "structured_info": {
            "start_date": "2025-08-10",
            "end_date": "2025-08-11"
        }
    }
    
    classifier = WeatherClassifier()
    
    # 模拟不同天气场景
    weather_scenarios = [
        {"name": "晴好天气", "weather": [{"fxDate": "2025-08-10", "textDay": "晴"}, {"fxDate": "2025-08-11", "textDay": "多云"}]},
        {"name": "降水天气", "weather": [{"fxDate": "2025-08-10", "textDay": "小雨"}, {"fxDate": "2025-08-11", "textDay": "中雨"}]},
        {"name": "极端天气", "weather": [{"fxDate": "2025-08-10", "textDay": "大风"}, {"fxDate": "2025-08-11", "textDay": "沙尘暴"}]},
    ]
    
    for scenario in weather_scenarios:
        print(f"\n🌤️ 场景: {scenario['name']}")
        
        # 分析天气
        trip_dates = ["2025-08-10", "2025-08-11"]
        weather_analysis = classifier.analyze_trip_weather(scenario['weather'], trip_dates)
        
        print(f"天气评估: {weather_analysis['overall_assessment']}")
        
        # 模拟景点筛选逻辑
        candidate_pois = test_state["candidate_pois"]
        
        # 统计天气类型
        indoor_days = sum(1 for day in weather_analysis["daily_weather"] 
                         if day["suitability"].value == "可市内出行")
        outdoor_days = sum(1 for day in weather_analysis["daily_weather"] 
                          if day["suitability"].value == "可户外出行")
        bad_days = sum(1 for day in weather_analysis["daily_weather"] 
                      if day["suitability"].value == "不建议出行")
        
        if bad_days > len(trip_dates) // 2:
            filtered_pois = []
            reason = "天气恶劣"
        elif indoor_days > outdoor_days:
            filtered_pois = classifier.get_suitable_pois(candidate_pois, 
                                                       classifier.classify_weather("中雨"))
            reason = "降水天气，选择室内景点"
        else:
            filtered_pois = candidate_pois
            reason = "天气良好，保留所有景点"
        
        print(f"筛选策略: {reason}")
        print(f"筛选结果: {len(candidate_pois)} → {len(filtered_pois)} 个景点")
        
        if filtered_pois:
            print("保留景点:")
            for poi in filtered_pois:
                print(f"  ✓ {poi['name']}")

def test_new_weather_constraint_logic():
    """测试新的天气约束流程逻辑"""
    
    print("\n🔄 测试新的天气约束流程逻辑")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 测试景点数据
    test_pois = [
        {"name": "故宫博物院", "indoor": True, "suggested_duration_hours": 3.0},
        {"name": "八达岭长城", "indoor": False, "suggested_duration_hours": 4.0},
        {"name": "天坛", "indoor": False, "suggested_duration_hours": 2.5},
        {"name": "国家博物馆", "indoor": True, "suggested_duration_hours": 2.0},
        {"name": "颐和园", "indoor": False, "suggested_duration_hours": 3.5},
        {"name": "中国科技馆", "indoor": True, "suggested_duration_hours": 2.5},
    ]
    
    # 测试场景1：极端天气阻断出行
    print("\n📊 测试场景1: 极端天气阻断出行")
    extreme_weather_data = [
        {"fxDate": "2025-08-10", "textDay": "台风", "tempMax": "25", "tempMin": "18", "precip": "0.0"},
        {"fxDate": "2025-08-11", "textDay": "大风", "tempMax": "22", "tempMin": "15", "precip": "0.0"},
    ]
    trip_dates = ["2025-08-10", "2025-08-11"]
    weather_analysis = classifier.analyze_trip_weather(extreme_weather_data, trip_dates)
    
    is_blocked = classifier.check_extreme_weather_blocking(weather_analysis, len(trip_dates))
    print(f"极端天气天数: {weather_analysis['extreme_weather_days']}")
    print(f"总行程天数: {len(trip_dates)}")
    print(f"是否被阻断: {'是' if is_blocked else '否'}")
    
    # 测试场景2：必去景点天气冲突
    print("\n📊 测试场景2: 必去景点天气冲突")
    rain_weather_data = [
        {"fxDate": "2025-08-10", "textDay": "大雨", "tempMax": "25", "tempMin": "18", "precip": "15.0"},
        {"fxDate": "2025-08-11", "textDay": "中雨", "tempMax": "22", "tempMin": "15", "precip": "8.0"},
    ]
    weather_analysis_rain = classifier.analyze_trip_weather(rain_weather_data, trip_dates)
    
    # 假设八达岭长城是必去景点
    must_visit_pois = [{"name": "八达岭长城", "indoor": False}]
    has_conflict = classifier.check_must_visit_weather_conflict(weather_analysis_rain, must_visit_pois)
    print(f"降水天数: {weather_analysis_rain['indoor_days']}")
    print(f"户外适宜天数: {weather_analysis_rain['suitable_days']}")
    print(f"必去景点: {[poi['name'] for poi in must_visit_pois]}")
    print(f"是否有冲突: {'是' if has_conflict else '否'}")
    
    # 测试场景3：景点筛选
    print("\n📊 测试场景3: 景点筛选")
    filtered_pois = classifier.filter_completely_inaccessible_pois(test_pois, weather_analysis_rain)
    print(f"原始景点数: {len(test_pois)}")
    print(f"筛选后景点数: {len(filtered_pois)}")
    print("保留的景点:")
    for poi in filtered_pois:
        indoor_status = poi.get("indoor", "未知")
        print(f"  ✓ {poi['name']} (室内: {indoor_status})")
    
    # 测试场景4：行程饱满度检查
    print("\n📊 测试场景4: 行程饱满度检查")
    daily_time_budget = 12  # 12小时/天
    trip_days = 2
    
    # 测试饱满的行程
    full_pois = test_pois  # 所有景点总时长约17.5小时，2天24小时，差值6.5小时 < 10，饱满
    is_full, analysis = classifier.check_trip_fullness(full_pois, daily_time_budget, trip_days)
    print(f"全景点行程 - 总时间预算: {analysis['total_time_budget']}h")
    print(f"全景点行程 - 景点总时长: {analysis['total_suggested_hours']}h") 
    print(f"全景点行程 - 时间差: {analysis['time_difference']}h")
    print(f"全景点行程 - 是否饱满: {'是' if is_full else '否'}")
    
    # 测试不饱满的行程
    sparse_pois = test_pois[:2]  # 只有前2个景点，总时长5小时，差值19小时 > 10，不饱满
    is_full_sparse, analysis_sparse = classifier.check_trip_fullness(sparse_pois, daily_time_budget, trip_days)
    print(f"稀疏行程 - 总时间预算: {analysis_sparse['total_time_budget']}h")
    print(f"稀疏行程 - 景点总时长: {analysis_sparse['total_suggested_hours']}h")
    print(f"稀疏行程 - 时间差: {analysis_sparse['time_difference']}h")
    print(f"稀疏行程 - 是否饱满: {'是' if is_full_sparse else '否'}")

def test_weather_constraint_flow_integration():
    """测试完整的天气约束流程集成"""
    
    print("\n🔗 测试完整的天气约束流程集成")
    print("=" * 50)
    
    classifier = WeatherClassifier()
    
    # 模拟完整的测试数据
    test_state = {
        "candidate_pois": [
            {"name": "故宫博物院", "indoor": True, "suggested_duration_hours": 3.0},
            {"name": "八达岭长城", "indoor": False, "suggested_duration_hours": 4.0},
            {"name": "天坛", "indoor": False, "suggested_duration_hours": 2.5},
            {"name": "国家博物馆", "indoor": True, "suggested_duration_hours": 2.0},
            {"name": "颐和园", "indoor": False, "suggested_duration_hours": 3.5},
            {"name": "中国科技馆", "indoor": True, "suggested_duration_hours": 2.5},
        ],
        "structured_info": {
            "start_date": "2025-08-10",
            "end_date": "2025-08-11",
            "preferences": {
                "must_visit": ["八达岭长城"]
            },
            "constraints": {
                "derived": {
                    "daily_time_budget_hours": 12
                }
            }
        }
    }
    
    # 测试场景：好天气，应该通过所有检查
    good_weather_data = [
        {"fxDate": "2025-08-10", "textDay": "晴", "tempMax": "28", "tempMin": "18", "precip": "0.0"},
        {"fxDate": "2025-08-11", "textDay": "多云", "tempMax": "26", "tempMin": "16", "precip": "0.0"},
    ]
    
    print("\n场景：好天气流程测试")
    trip_dates = ["2025-08-10", "2025-08-11"]
    weather_analysis = classifier.analyze_trip_weather(good_weather_data, trip_dates)
    
    # A. 极端天气检查
    is_blocked = classifier.check_extreme_weather_blocking(weather_analysis, len(trip_dates))
    print(f"A. 极端天气检查: {'❌ 阻断' if is_blocked else '✅ 通过'}")
    
    if not is_blocked:
        # B. 必去景点检查  
        must_visit_pois = [{"name": "八达岭长城", "indoor": False}]
        has_conflict = classifier.check_must_visit_weather_conflict(weather_analysis, must_visit_pois)
        print(f"B. 必去景点检查: {'❌ 冲突' if has_conflict else '✅ 通过'}")
        
        if not has_conflict:
            # C. 景点筛选
            filtered_pois = classifier.filter_completely_inaccessible_pois(test_state["candidate_pois"], weather_analysis)
            print(f"C. 景点筛选: {len(test_state['candidate_pois'])} → {len(filtered_pois)} 个景点")
            
            # D. 饱满度检查
            is_full, analysis = classifier.check_trip_fullness(filtered_pois, 12, 2)
            print(f"D. 饱满度检查: {'✅ 饱满' if is_full else '❌ 不饱满'} (差值: {analysis['time_difference']}h)")
            
            if is_full:
                print("🎉 天气约束流程全部通过！")
            else:
                print("⚠️ 行程不够饱满，需要重新安排")
        else:
            print("⚠️ 必去景点受天气影响，需要重新安排")
    else:
        print("⚠️ 极端天气阻断出行，需要重新安排")

if __name__ == "__main__":
    print("🧪 weather_filter 节点测试套件")
    print("=" * 60)
    
    # 测试天气分类器
    classifier = test_weather_classifier()
    
    # 测试景点筛选
    test_poi_filtering()
    
    # 测试天气分析
    test_trip_weather_analysis()
    
    # 模拟节点测试
    test_weather_filter_node_simulation()
    
    # 测试新的天气约束逻辑
    test_new_weather_constraint_logic()
    
    # 测试完整流程集成
    test_weather_constraint_flow_integration()
    
    print("\n🎉 测试完成!")
    print("\n💡 使用说明:")
    print("1. 需要在 .env 文件中设置 HEFENG_API_HOST 和 HEFENG_API_KEY")
    print("2. 实际运行时会调用和风天气API获取真实天气数据")
    print("3. 新的天气约束流程包含4个步骤：")
    print("   A. 极端天气阻断检查")
    print("   B. 必去景点天气冲突检查") 
    print("   C. 完全不可访问景点筛选")
    print("   D. 行程饱满度检查")
    print("4. 任一步骤失败都会建议用户重新选择日期")
