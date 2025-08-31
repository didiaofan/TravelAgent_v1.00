#!/usr/bin/env python3
"""
调试intensity_calculate节点的问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_calculate_plan_intensity_simple(daily_candidates, transport_plan):
    """调试版本的计算函数"""
    
    print(f"\n=== 调试 _calculate_plan_intensity_simple ===")
    
    plan_strategy = transport_plan.get("strategy", "未知方案")
    daily_plans = transport_plan.get("daily_plans", [])
    
    print(f"计划策略: {plan_strategy}")
    print(f"每日计划数量: {len(daily_plans)}")
    print(f"候选日程数量: {len(daily_candidates)}")
    
    result = {
        "strategy": plan_strategy,
        "daily_details": [],
        "total_poi_hours": 0,
        "total_transport_hours": 0,
        "total_hours": 0,
        "avg_daily_hours": 0
    }
    
    for i, transport_day in enumerate(daily_plans):
        day_idx = transport_day.get("day", i + 1)
        print(f"\n处理第{i}个交通日计划，day_idx={day_idx}")
        print(f"transport_day = {transport_day}")
        
        # 获取对应的景点安排
        poi_day = None
        for candidate_day in daily_candidates:
            candidate_day_idx = candidate_day.get("day", 0)
            print(f"  检查候选日程: day={candidate_day_idx}")
            if candidate_day_idx == day_idx:
                poi_day = candidate_day
                print(f"  ✅ 找到匹配的POI日程: {poi_day}")
                break
        
        if not poi_day:
            print(f"  ❌ 未找到第{day_idx}天的POI安排，跳过")
            continue
            
        # 计算景点游玩时间（小时）
        poi_hours = 0
        pois = poi_day.get("pois", [])
        print(f"  景点列表: {len(pois)}个")
        for poi in pois:
            poi_duration = poi.get("suggested_duration_hours", 2.0)
            poi_name = poi.get("name", "未知")
            print(f"    {poi_name}: {poi_duration}小时")
            poi_hours += poi_duration
        
        # 计算交通时间（分钟转小时）
        transport_minutes = transport_day.get("day_total_time", 0)
        transport_hours = transport_minutes / 60.0
        
        print(f"  总景点时间: {poi_hours}小时")
        print(f"  交通时间: {transport_minutes}分钟 = {transport_hours}小时")
        
        # 每日总时间
        daily_total = poi_hours + transport_hours
        print(f"  每日总时间: {daily_total}小时")
        
        day_detail = {
            "day": day_idx,
            "date": poi_day.get("date", f"第{day_idx}天"),
            "poi_hours": poi_hours,
            "transport_hours": transport_hours,
            "total_hours": daily_total,
            "poi_count": len(poi_day.get("pois", [])),
            "poi_names": [poi.get("name", "") for poi in poi_day.get("pois", [])]
        }
        
        result["daily_details"].append(day_detail)
        result["total_poi_hours"] += poi_hours
        result["total_transport_hours"] += transport_hours
        result["total_hours"] += daily_total
    
    # 计算平均值
    if result["daily_details"]:
        result["avg_daily_hours"] = result["total_hours"] / len(result["daily_details"])
    
    print(f"\n最终结果:")
    print(f"  总POI时间: {result['total_poi_hours']}")
    print(f"  总交通时间: {result['total_transport_hours']}")
    print(f"  总时间: {result['total_hours']}")
    print(f"  平均每日: {result['avg_daily_hours']}")
    
    return result

def test_debug():
    """测试调试版本"""
    
    # 创建测试状态
    daily_candidates = [
        {
            "day": 1,
            "date": "2025-08-23",
            "pois": [
                {
                    "name": "故宫",
                    "suggested_duration_hours": 3.0
                },
                {
                    "name": "天安门广场", 
                    "suggested_duration_hours": 1.5
                }
            ]
        },
        {
            "day": 2,
            "date": "2025-08-24", 
            "pois": [
                {
                    "name": "北京环球影城",
                    "suggested_duration_hours": 8.0
                }
            ]
        }
    ]
    
    transport_plan = {
        "strategy": "最省时间",
        "daily_plans": [
            {
                "day": 1,
                "date": "2025-08-23",
                "day_total_time": 45.0,  # 45分钟
                "routes": []
            },
            {
                "day": 2,
                "date": "2025-08-24",
                "day_total_time": 60.0,  # 60分钟
                "routes": []
            }
        ]
    }
    
    print("=== 调试测试 ===")
    print(f"输入daily_candidates: {daily_candidates}")
    print(f"输入transport_plan: {transport_plan}")
    
    result = debug_calculate_plan_intensity_simple(daily_candidates, transport_plan)
    return result

if __name__ == "__main__":
    test_debug()

