#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的景点聚类算法
整合天数、天气、距离、个人偏好的智能每日行程分配
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
# 先用简单的距离计算替代geopy，避免依赖问题
# from geopy.distance import geodesic
import math

def calculate_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    使用haversine公式计算两点间距离（公里）
    """
    # 转换为弧度
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # haversine公式
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # 地球半径（公里）
    r = 6371
    return c * r

def improved_scenic_spots_clustering(state: dict) -> dict:
    """
    改进的景点聚类：整合所有约束条件的智能每日行程分配
    
    核心思路：
    1. 必去景点优先分配
    2. 基于真实地理距离聚类 
    3. 考虑天气约束
    4. 控制每日时间预算（预留交通时间）
    5. 最终输出确定的每日行程
    
    重要设计：
    - 每日总时间 = 景点游玩时间 + 交通时间（最多3.5小时）
    - 成年人：12小时总时间 → 8.5小时景点时间
    - 有老人/儿童：9小时总时间 → 5.5小时景点时间
    
    Args:
        state: Agent状态，包含weather_adjusted_pois, daily_available_pois等
        
    Returns:
        state: 更新后的状态，包含final_daily_itinerary
    """
    print("🎯 执行改进的景点聚类...")
    
    # 获取输入数据
    weather_adjusted_pois = state.get("weather_adjusted_pois", [])
    daily_available_pois = state.get("daily_available_pois", [])
    info = state.get("structured_info", {})
    preferences = info.get("preferences", {})
    must_visit_pois = preferences.get("must_visit", [])
    daily_time_budget = state.get("daily_time_limit", 12)
    
    # 🚗 关键：计算景点可用时间（预留交通时间）
    max_transport_time = 2.5  # 每日最大交通时间（小时）
    daily_poi_time_budget = daily_time_budget - max_transport_time
    
    print(f"⏰ 时间预算分配:")
    print(f"  每日总时间预算: {daily_time_budget}小时")
    print(f"  预留交通时间: {max_transport_time}小时")
    print(f"  景点可用时间: {daily_poi_time_budget}小时")
    
    # 获取行程日期
    start_date = info.get("start_date")
    end_date = info.get("end_date")
    
    if not start_date or not end_date or not weather_adjusted_pois:
        print("❌ 缺少必要数据，使用默认分配")
        state["daily_candidates"] = []
        return state
    
    # 计算行程天数和日期列表
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    trip_days = (end_date_obj - start_date_obj).days + 1
    
    trip_dates = []
    current_date = start_date_obj
    for i in range(trip_days):
        trip_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    print(f"📅 行程规划: {trip_days}天 ({start_date} 至 {end_date})")
    print(f"🏛️ 候选景点: {len(weather_adjusted_pois)}个")
    print(f"🎯 必去景点: {must_visit_pois}")
    
    # 执行多阶段智能分配（使用景点可用时间）
    final_itinerary = multi_stage_poi_allocation(
        weather_adjusted_pois=weather_adjusted_pois,
        daily_available_pois=daily_available_pois,
        trip_dates=trip_dates,
        must_visit_pois=must_visit_pois,
        daily_poi_time_budget=daily_poi_time_budget,  # 使用景点时间预算
        daily_total_time_budget=daily_time_budget     # 保留总时间用于最终输出
    )
    
    # 更新状态
    state["daily_candidates"] = final_itinerary
    
    # 检查必去景点安排情况
    check_must_visit_arrangement(final_itinerary, must_visit_pois)
    
    # 输出结果摘要
    print(f"\n🎉 每日行程分配完成！")
    total_pois = 0
    total_hours = 0
    
    for day_plan in final_itinerary:
        day_pois = len(day_plan["pois"])
        day_hours = day_plan.get("poi_hours", 0)  # 使用poi_hours字段
        total_pois += day_pois
        total_hours += day_hours
        
        print(f"  {day_plan['date']}: {day_pois}个景点, {day_hours:.1f}小时")
        for poi in day_plan["pois"]:
            hours = poi.get("suggested_duration_hours", 2.0)
            print(f"    - {poi['name']} ({hours}h)")
    
    print(f"  总计: {total_pois}个景点, {total_hours:.1f}小时")
    
    return state


def check_must_visit_arrangement(final_itinerary: List[Dict], must_visit_pois: List[str]) -> None:
    """检查必去景点是否都被安排，并输出提示"""
    
    if not must_visit_pois:
        return
    
    # 收集所有已安排的景点名称
    arranged_poi_names = set()
    for day_plan in final_itinerary:
        for poi in day_plan["pois"]:
            arranged_poi_names.add(poi["name"])
    
    # 检查每个必去景点
    arranged_must_visit = []
    missing_must_visit = []
    
    for must_visit_name in must_visit_pois:
        found = False
        for arranged_name in arranged_poi_names:
            # 使用包含关系检查，允许部分匹配
            if (must_visit_name.lower() in arranged_name.lower() or 
                arranged_name.lower() in must_visit_name.lower()):
                arranged_must_visit.append((must_visit_name, arranged_name))
                found = True
                break
        
        if not found:
            missing_must_visit.append(must_visit_name)
    
    # 输出检查结果
    print(f"\n🎯 必去景点安排检查:")
    
    if arranged_must_visit:
        print(f"  ✅ 已安排的必去景点:")
        for requested, arranged in arranged_must_visit:
            print(f"    - {requested} → {arranged}")
    
    if missing_must_visit:
        print(f"  ❌ 未能安排的必去景点:")
        for missing in missing_must_visit:
            print(f"    - {missing}")
        
        print(f"\n💡 未安排原因可能包括:")
        print(f"  1. 天气约束：景点不适合行程期间的天气条件")
        print(f"  2. 时间限制：景点游玩时间超出每日预算(2.5h交通+景点时间)")
        print(f"  3. 地理位置：景点距离其他景点过远，难以合理安排")
        print(f"  4. 景点信息：在景点数据库中未找到对应景点")
        
        print(f"\n🔧 建议解决方案:")
        print(f"  1. 延长行程天数以增加时间预算")
        print(f"  2. 调整其他景点选择，为必去景点腾出时间")
        print(f"  3. 选择距离更近的住宿位置减少交通时间")
        print(f"  4. 考虑将大型景点(如环球影城)单独安排一天")
    else:
        print(f"  🎉 所有必去景点都已成功安排！")


def multi_stage_poi_allocation(
    weather_adjusted_pois: List[Dict],
    daily_available_pois: List[Dict], 
    trip_dates: List[str],
    must_visit_pois: List[str],
    daily_poi_time_budget: float,      # 景点可用时间
    daily_total_time_budget: float     # 总时间预算
) -> List[Dict]:
    """
    多阶段景点分配策略
    
    前置条件: weather_adjusted_pois已通过weather_filter节点的天气筛选
    
    时间预算说明:
    - daily_poi_time_budget: 景点游玩可用时间（已扣除交通时间3.5h）
    - daily_total_time_budget: 每日总时间预算（用于最终报告）
    
    阶段1: 必去景点优先分配
    阶段2: 高时间消耗景点独立分配 
    阶段3: 地理距离聚类
    阶段4: 天气约束验证（依赖前期处理结果）
    阶段5: 时间预算平衡
    """
    
    print(f"🎯 多阶段分配开始 (景点时间预算: {daily_poi_time_budget}h, 总时间预算: {daily_total_time_budget}h)")
    
    # 阶段1: 识别和预分配必去景点
    print("\n📍 阶段1: 必去景点优先分配")
    must_visit_allocation = allocate_must_visit_pois(
        weather_adjusted_pois, trip_dates, must_visit_pois, daily_poi_time_budget
    )
    
    # 阶段2: 处理高时间消耗景点（如环球影城）
    print("\n⏰ 阶段2: 高时间消耗景点处理")
    high_time_allocation = handle_high_time_pois(
        must_visit_allocation, daily_poi_time_budget
    )
    
    # 阶段3: 剩余景点地理距离聚类
    print("\n🗺️ 阶段3: 剩余景点地理聚类")
    geographic_allocation = geographic_clustering_remaining(
        high_time_allocation, weather_adjusted_pois, trip_dates
    )
    
    # 阶段4: 天气约束优化
    print("\n🌤️ 阶段4: 天气约束优化")
    weather_optimized = optimize_for_weather(
        geographic_allocation, daily_available_pois
    )
    
    # 阶段5: 最终时间预算平衡
    print("\n⚖️ 阶段5: 时间预算平衡")
    final_allocation = balance_time_budget(
        weather_optimized, daily_poi_time_budget, daily_total_time_budget
    )
    
    return final_allocation


def allocate_must_visit_pois(
    weather_adjusted_pois: List[Dict],
    trip_dates: List[str], 
    must_visit_pois: List[str],
    daily_poi_time_budget: float
) -> List[Dict]:
    """阶段1: 优先分配必去景点"""
    
    # 初始化每日计划
    daily_plans = []
    for date in trip_dates:
        daily_plans.append({
            "date": date,
            "pois": [],
            "allocated_hours": 0,
            "remaining_capacity": daily_poi_time_budget  # 使用景点时间预算
        })
    
    # 找到必去景点对象
    must_visit_poi_objects = []
    for must_name in must_visit_pois:
        for poi in weather_adjusted_pois:
            if (must_name.lower() in poi.get("name", "").lower() or 
                poi.get("name", "").lower() in must_name.lower()):
                must_visit_poi_objects.append(poi)
                break
    
    print(f"  找到必去景点: {[poi['name'] for poi in must_visit_poi_objects]}")
    
    # 按时间消耗排序（高时间消耗优先）
    must_visit_poi_objects.sort(
        key=lambda x: x.get("suggested_duration_hours", 2.0), 
        reverse=True
    )
    
    # 分配必去景点
    for poi in must_visit_poi_objects:
        poi_hours = poi.get("suggested_duration_hours", 2.0)
        
        # 找到最适合的一天
        best_day_idx = None
        min_waste = float('inf')
        
        for i, day_plan in enumerate(daily_plans):
            if day_plan["remaining_capacity"] >= poi_hours:
                waste = day_plan["remaining_capacity"] - poi_hours
                if waste < min_waste:
                    min_waste = waste
                    best_day_idx = i
        
        # 分配到最佳一天
        if best_day_idx is not None:
            daily_plans[best_day_idx]["pois"].append(poi)
            daily_plans[best_day_idx]["allocated_hours"] += poi_hours
            daily_plans[best_day_idx]["remaining_capacity"] -= poi_hours
            print(f"  ✅ {poi['name']} → 第{best_day_idx+1}天 ({poi_hours}h)")
        else:
            print(f"  ⚠️ {poi['name']} 无法安排 (需要{poi_hours}h)")
    
    return daily_plans


def handle_high_time_pois(
    daily_plans: List[Dict],
    daily_poi_time_budget: float
) -> List[Dict]:
    """阶段2: 处理高时间消耗景点"""
    
    high_time_threshold = daily_poi_time_budget * 0.6  # 超过60%算高时间消耗
    
    for day_plan in daily_plans:
        high_time_pois = [
            poi for poi in day_plan["pois"] 
            if poi.get("suggested_duration_hours", 2.0) >= high_time_threshold
        ]
        
        if high_time_pois:
            poi_name = high_time_pois[0]["name"]
            hours = high_time_pois[0].get("suggested_duration_hours", 2.0)
            print(f"  🎢 {poi_name} 为高时间消耗景点 ({hours}h)，建议独立安排")
            
            # 如果一天有多个高时间消耗景点，提示用户
            if len(high_time_pois) > 1:
                print(f"  ⚠️ 第{daily_plans.index(day_plan)+1}天有多个高时间消耗景点，可能过于紧张")
    
    return daily_plans


def geographic_clustering_remaining(
    daily_plans: List[Dict],
    all_pois: List[Dict],
    trip_dates: List[str]
) -> List[Dict]:
    """阶段3: 对剩余景点进行地理聚类"""
    
    # 找出已分配的景点
    allocated_poi_names = set()
    for day_plan in daily_plans:
        for poi in day_plan["pois"]:
            allocated_poi_names.add(poi["name"])
    
    # 剩余景点
    remaining_pois = [
        poi for poi in all_pois 
        if poi["name"] not in allocated_poi_names
    ]
    
    print(f"  剩余景点数量: {len(remaining_pois)}")
    
    if not remaining_pois:
        return daily_plans
    
    # 创建全局已使用景点记录，避免重复分配
    global_used_pois = set(allocated_poi_names)
    print(f"  已分配景点: {list(global_used_pois)}")
    
    # 为有剩余容量的天数分配景点
    for i, day_plan in enumerate(daily_plans):
        if day_plan["remaining_capacity"] > 2:  # 至少2小时剩余容量
            
            # 添加day_index以便调试
            day_plan["day_index"] = i + 1
            
            # 过滤掉已经被其他天使用的景点
            available_pois = [
                poi for poi in remaining_pois 
                if poi["name"] not in global_used_pois
            ]
            
            print(f"    第{i+1}天可用景点: {len(available_pois)}个 (剩余容量: {day_plan['remaining_capacity']:.1f}h)")
            
            # 如果这一天已有景点，基于地理位置选择近距离景点
            if day_plan["pois"]:
                nearby_pois = find_nearby_pois(
                    day_plan["pois"], available_pois, max_distance_km=15
                )
            else:
                nearby_pois = available_pois[:5]  # 取前5个高评分景点
            
            # 填充当天剩余时间，并更新全局使用记录
            fill_remaining_time(day_plan, nearby_pois, available_pois, global_used_pois)
    
    return daily_plans


def find_nearby_pois(
    existing_pois: List[Dict], 
    candidate_pois: List[Dict],
    max_distance_km: float = 15
) -> List[Dict]:
    """找到与已有景点距离较近的候选景点"""
    
    if not existing_pois:
        return candidate_pois
    
    # 计算已有景点的中心点
    center_lat = np.mean([poi.get("lat", 39.9042) for poi in existing_pois])
    center_lng = np.mean([poi.get("lng", 116.4074) for poi in existing_pois])
    
    # 筛选距离中心点较近的景点
    nearby_pois = []
    for poi in candidate_pois:
        poi_lat = poi.get("lat", 39.9042)
        poi_lng = poi.get("lng", 116.4074)
        
        # 使用简化的距离计算（haversine公式）
        distance = calculate_distance_km(center_lat, center_lng, poi_lat, poi_lng)
        
        if distance <= max_distance_km:
            poi_with_distance = poi.copy()
            poi_with_distance["distance_to_center"] = distance
            nearby_pois.append(poi_with_distance)
    
    # 按距离排序
    nearby_pois.sort(key=lambda x: x["distance_to_center"])
    
    return nearby_pois


def fill_remaining_time(
    day_plan: Dict,
    preferred_pois: List[Dict],
    all_remaining_pois: List[Dict],
    global_used_pois: set
) -> None:
    """填充当天剩余时间（带全局去重）"""
    
    # 优先使用附近景点，但要确保未被全局使用
    candidate_pool = [
        poi for poi in preferred_pois 
        if poi["name"] not in global_used_pois
    ] + [
        poi for poi in all_remaining_pois 
        if poi not in preferred_pois and poi["name"] not in global_used_pois
    ]
    
    # 按评分排序
    candidate_pool.sort(
        key=lambda x: x.get("score", x.get("popularity_score", 0.5)), 
        reverse=True
    )
    
    for poi in candidate_pool:
        poi_hours = poi.get("suggested_duration_hours", 2.0)
        
        # 检查时间和全局去重
        if (day_plan["remaining_capacity"] >= poi_hours and 
            poi["name"] not in global_used_pois):
            
            day_plan["pois"].append(poi)
            day_plan["allocated_hours"] += poi_hours
            day_plan["remaining_capacity"] -= poi_hours
            
            # 关键：添加到全局已使用集合
            global_used_pois.add(poi["name"])
            
            print(f"  ➕ 添加 {poi['name']} → 第{day_plan.get('day_index', '?')}天 ({poi_hours}h)")
            
            # 如果剩余时间不足2小时，停止添加
            if day_plan["remaining_capacity"] < 2:
                break


def optimize_for_weather(
    daily_plans: List[Dict],
    daily_available_pois: List[Dict]
) -> List[Dict]:
    """阶段4: 天气约束优化（简化版 - 依赖前期weather_filter节点的处理结果）"""
    
    print("  ✅ 天气约束已在weather_filter节点处理完成")
    print("  ℹ️ 输入的weather_adjusted_pois已经过天气筛选，无需重复检查")
    
    # 由于weather_adjusted_pois已经是经过天气筛选的结果，
    # 而且用户要求景点选择不再调整，这里直接返回
    return daily_plans


def balance_time_budget(
    daily_plans: List[Dict],
    daily_poi_time_budget: float,      # 景点时间预算
    daily_total_time_budget: float     # 总时间预算
) -> List[Dict]:
    """阶段5: 智能时间预算平衡"""
    
    print("  🔄 开始智能时间预算平衡...")
    
    # 定义时间利用率的理想范围（基于景点时间）
    optimal_min = 0.6  # 最低60%利用率
    optimal_max = 0.9  # 最高90%利用率
    
    # 第一步：分析每日时间分布
    time_analysis = analyze_daily_time_distribution(daily_plans, daily_poi_time_budget)
    
    # 第二步：识别需要调整的天数
    adjustment_plan = identify_time_imbalances(time_analysis, optimal_min, optimal_max)
    
    # 第三步：执行时间平衡调整
    balanced_plans = execute_time_balancing(daily_plans, adjustment_plan, daily_poi_time_budget)
    
    # 第四步：生成最终行程格式
    final_itinerary = format_final_itinerary(balanced_plans, daily_poi_time_budget, daily_total_time_budget)
    
    # 输出平衡结果摘要
    print_balance_summary(final_itinerary, daily_poi_time_budget, daily_total_time_budget)
    
    return final_itinerary


def analyze_daily_time_distribution(
    daily_plans: List[Dict], 
    daily_time_budget: float
) -> List[Dict]:
    """分析每日时间分布"""
    
    analysis = []
    
    for i, day_plan in enumerate(daily_plans):
        total_hours = sum(
            poi.get("suggested_duration_hours", 2.0) 
            for poi in day_plan["pois"]
        )
        
        utilization = total_hours / daily_time_budget if daily_time_budget > 0 else 0
        remaining_time = daily_time_budget - total_hours
        
        day_analysis = {
            "day_index": i,
            "date": day_plan["date"],
            "pois": day_plan["pois"],
            "total_hours": total_hours,
            "utilization": utilization,
            "remaining_time": remaining_time,
            "status": get_time_status(utilization)
        }
        
        analysis.append(day_analysis)
        
        print(f"    第{i+1}天 ({day_plan['date']}): {total_hours:.1f}h/{daily_time_budget}h ({utilization*100:.1f}%) - {day_analysis['status']}")
    
    return analysis


def get_time_status(utilization: float) -> str:
    """获取时间利用状态"""
    if utilization < 0.5:
        return "时间过少"
    elif utilization < 0.6:
        return "略显空闲"
    elif utilization <= 0.9:
        return "时间合理"
    elif utilization <= 1.1:
        return "略显紧张"
    else:
        return "时间过多"


def identify_time_imbalances(
    time_analysis: List[Dict], 
    optimal_min: float, 
    optimal_max: float
) -> Dict:
    """识别需要调整的时间不平衡"""
    
    over_time_days = []  # 时间过多的天数
    under_time_days = []  # 时间过少的天数
    
    for day_analysis in time_analysis:
        utilization = day_analysis["utilization"]
        
        if utilization > optimal_max:
            over_time_days.append(day_analysis)
        elif utilization < optimal_min:
            under_time_days.append(day_analysis)
    
    adjustment_plan = {
        "over_time_days": over_time_days,
        "under_time_days": under_time_days,
        "needs_adjustment": len(over_time_days) > 0 or len(under_time_days) > 0
    }
    
    if adjustment_plan["needs_adjustment"]:
        print(f"  📋 识别到时间不平衡: {len(over_time_days)}天过多, {len(under_time_days)}天过少")
    else:
        print(f"  ✅ 时间分布合理，无需调整")
    
    return adjustment_plan


def execute_time_balancing(
    daily_plans: List[Dict], 
    adjustment_plan: Dict,
    daily_time_budget: float
) -> List[Dict]:
    """执行时间平衡调整（保守策略）"""
    
    if not adjustment_plan["needs_adjustment"]:
        return daily_plans
    
    print("  🔧 执行时间平衡调整...")
    
    # 由于用户要求景点选择不再调整，这里采用保守策略
    # 主要是记录建议和警告，而不是强制调整景点
    
    balanced_plans = []
    
    for day_plan in daily_plans:
        balanced_day = day_plan.copy()
        
        # 计算当天状态
        total_hours = sum(
            poi.get("suggested_duration_hours", 2.0) 
            for poi in day_plan["pois"]
        )
        utilization = total_hours / daily_time_budget
        
        # 添加调整建议
        if utilization > 0.9:
            balanced_day["adjustment_suggestion"] = "建议缩短部分景点游玩时间或考虑分散到其他天"
            balanced_day["adjustment_type"] = "reduce_time"
        elif utilization < 0.6:
            balanced_day["adjustment_suggestion"] = "可以增加更多景点或延长现有景点的游玩时间"
            balanced_day["adjustment_type"] = "add_time"
        else:
            balanced_day["adjustment_suggestion"] = "时间安排合理"
            balanced_day["adjustment_type"] = "optimal"
        
        balanced_plans.append(balanced_day)
    
    return balanced_plans


def format_final_itinerary(
    balanced_plans: List[Dict], 
    daily_poi_time_budget: float,      # 景点时间预算
    daily_total_time_budget: float     # 总时间预算
) -> List[Dict]:
    """格式化最终行程"""
    
    final_itinerary = []
    transport_time_reserved = daily_total_time_budget - daily_poi_time_budget
    
    for day_plan in balanced_plans:
        poi_hours = sum(
            poi.get("suggested_duration_hours", 2.0) 
            for poi in day_plan["pois"]
        )
        
        # 计算总时间（景点时间 + 预留交通时间）
        estimated_total_hours = poi_hours + transport_time_reserved
        
        # 利用率基于景点时间预算计算
        poi_utilization = poi_hours / daily_poi_time_budget if daily_poi_time_budget > 0 else 0
        
        # 总时间利用率
        total_utilization = estimated_total_hours / daily_total_time_budget if daily_total_time_budget > 0 else 0
        
        final_day = {
            "date": day_plan["date"],
            "pois": day_plan["pois"],
            "poi_hours": poi_hours,                          # 纯景点时间
            "transport_hours_reserved": transport_time_reserved,  # 预留交通时间
            "estimated_total_hours": estimated_total_hours,   # 预估总时间
            "poi_count": len(day_plan["pois"]),
            "poi_time_utilization": poi_utilization,          # 景点时间利用率
            "total_time_utilization": total_utilization,      # 总时间利用率
            "status": get_time_status(poi_utilization),
            "adjustment_suggestion": day_plan.get("adjustment_suggestion", ""),
            "adjustment_type": day_plan.get("adjustment_type", "optimal")
        }
        
        final_itinerary.append(final_day)
    
    return final_itinerary


def print_balance_summary(
    final_itinerary: List[Dict], 
    daily_poi_time_budget: float,
    daily_total_time_budget: float
):
    """输出平衡结果摘要"""
    
    total_pois = sum(day["poi_count"] for day in final_itinerary)
    total_poi_hours = sum(day["poi_hours"] for day in final_itinerary)
    total_estimated_hours = sum(day["estimated_total_hours"] for day in final_itinerary)
    avg_poi_utilization = np.mean([day["poi_time_utilization"] for day in final_itinerary])
    avg_total_utilization = np.mean([day["total_time_utilization"] for day in final_itinerary])
    
    optimal_days = len([day for day in final_itinerary if 0.6 <= day["poi_time_utilization"] <= 0.9])
    transport_reserved = daily_total_time_budget - daily_poi_time_budget
    
    print(f"\n  📊 时间预算平衡结果:")
    print(f"    总景点数: {total_pois}个")
    print(f"    景点游玩时长: {total_poi_hours:.1f}小时") 
    print(f"    预留交通时间: {transport_reserved}小时/天")
    print(f"    预估总时长: {total_estimated_hours:.1f}小时")
    print(f"    景点时间利用率: {avg_poi_utilization*100:.1f}%")
    print(f"    总时间利用率: {avg_total_utilization*100:.1f}%")
    print(f"    时间合理天数: {optimal_days}/{len(final_itinerary)}天")
    
    # 显示每日详细分配
    print(f"\n  📋 每日时间分配:")
    for day in final_itinerary:
        poi_pct = day["poi_time_utilization"] * 100
        total_pct = day["total_time_utilization"] * 100
        print(f"    {day['date']}: 景点{day['poi_hours']:.1f}h + 交通{transport_reserved}h = 总计{day['estimated_total_hours']:.1f}h ({total_pct:.1f}%)")
        
        # 显示需要注意的天数
        if day["adjustment_type"] != "optimal":
            print(f"      ⚠️ {day['adjustment_suggestion']}")
    
    print(f"\n  💡 后续节点说明:")
    print(f"    - hotel_selection: 选择酒店位置")
    print(f"    - transportation_planning: 计算实际交通时间")
    print(f"    - 如实际交通时间超过{transport_reserved}h，将优化酒店/交通方式，不回退景点选择")
