import json
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from config import config
from tools.routeinf import get_route_info

def _remove_json_comments(raw_text: str) -> str:
    """移除 // 行内注释，便于加载包含注释的 JSON。"""
    return re.sub(r"//.*", "", raw_text)

def load_poi_data(file_path: str = "data/beijing_poi.json") -> List[Dict[str, Any]]:
    """加载景点数据，支持含注释的 JSON。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = _remove_json_comments(raw)
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def determine_daily_time_budget(group: Optional[Dict[str, Any]]) -> int:
    """只有成人：10h；含老人/儿童：8h。"""
    if not group:
        return 12
    num_children = int((group.get("children") or 0))
    num_elderly = int((group.get("elderly") or 0))
    if num_children == 0 and num_elderly == 0:
        return 12
    return 8

def compute_trip_days(start_date: Optional[str], end_date: Optional[str]) -> int:
    """计算行程天数；若无效则返回 1。按自然日差计算（含起止两端）。"""
    try:
        if not start_date or not end_date:
            return 1
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        return max(1, (d2 - d1).days + 1)
    except Exception:
        return 1

def is_poi_suitable_for_group(poi: Dict[str, Any], group: Optional[Dict[str, Any]]) -> bool:
    """根据 `suitable_for` 与团队构成过滤。"""
    if not group:
        return True
    flags = set([str(s) for s in (poi.get("suitable_for") or [])])
    num_children = int((group.get("children") or 0))
    num_elderly = int((group.get("elderly") or 0))
    if num_children > 0 and not ("儿童" in flags or "家庭" in flags or "青少年" in flags):
        return False
    if num_elderly > 0 and ("老人" not in flags):
        return False
    return True

def compute_poi_score(poi: Dict[str, Any], preferences: Optional[Dict[str, Any]]) -> float:
    """综合 popularity_score 与偏好匹配得分。"""
    base = float(poi.get("popularity_score") or 0.0)
    if not preferences:
        return base
    
    # 处理避免列表：直接减分
    avoid_list = set([a.strip() for a in (preferences.get("avoid") or []) if str(a).strip()])
    poi_tags = set([str(t) for t in (poi.get("tags") or [])])
    if poi.get("name") in avoid_list or (avoid_list and (poi_tags & avoid_list)):
        base -= 1.0  # 避免景点减1分
    
    # 处理必去景点：大幅加分（支持模糊匹配）
    must_visit = set([m.strip() for m in (preferences.get("must_visit") or []) if str(m).strip()])
    poi_name = poi.get("name", "")
    
    # 精确匹配
    if poi_name in must_visit:
        base += 1.0  # 必去景点加1分
    else:
        # 模糊匹配：检查必去景点名称是否包含在POI名称中，或POI名称是否包含必去景点名称
        for must_visit_name in must_visit:
            if (must_visit_name in poi_name) or (poi_name in must_visit_name):
                base += 1.0  # 必去景点加1分
                break
    
    # 处理偏好类型：适度加分
    preferred_types = set([t.strip() for t in (preferences.get("attraction_types") or []) if str(t).strip()])
    if preferred_types and (poi_tags & preferred_types):
        base += 0.3  # 偏好类型加0.3分
    
    return base

def generate_preference_filtered_candidates(
    group: Dict[str, Any], 
    preferences: Dict[str, Any], 
    trip_days: int
) -> List[Dict[str, Any]]:
    """
    按偏好和受欢迎程度生成候选景点列表
    
    Args:
        group: 团队信息
        preferences: 用户偏好 
        trip_days: 游玩天数
        
    Returns:
        候选景点列表，按综合得分排序
    """
    import json
    import os
    
    # 确保每天至少4个候选景点
    min_candidates = trip_days * 4
    
    try:
        # 1. 读取景点数据（使用现有的注释处理函数）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        poi_file_path = os.path.join(current_dir, '..', 'data', 'beijing_poi.json')
        all_pois = load_poi_data(poi_file_path)
        
        # 2. 过滤适合团队的景点
        suitable_pois = []
        for poi in all_pois:
            if is_poi_suitable_for_group(poi, group):
                suitable_pois.append(poi)
        
        # 3. 计算综合得分并排序
        scored_pois = []
        for poi in suitable_pois:
            score = compute_poi_score(poi, preferences)
            poi_with_score = poi.copy()
            poi_with_score['computed_score'] = score
            scored_pois.append(poi_with_score)
        
        # 4. 按综合得分排序（必去景点和高评分景点优先）
        scored_pois.sort(key=lambda x: x['computed_score'], reverse=True)
        
        # 5. 选择候选景点（取足够数量，但不限制上限）
        # 至少选择 min_candidates 个，但如果有更多合适的也可以选择
        target_count = max(min_candidates, min(len(scored_pois), min_candidates ))  # 最多额外选6个
        final_candidates = scored_pois[:target_count]
        
        print(f"偏好筛选完成：从{len(all_pois)}个景点中筛选出{len(final_candidates)}个候选景点")
        print(f"游玩天数：{trip_days}天，最小候选数：{min_candidates}个")
        
        # 打印关键信息
        must_visit = set([m.strip() for m in (preferences.get("must_visit") or []) if str(m).strip()])
        if must_visit:
            # 检查哪些必去景点被包含（支持模糊匹配）
            found_must_visit = []
            for poi in final_candidates:
                poi_name = poi.get('name', '')
                # 精确匹配
                if poi_name in must_visit:
                    found_must_visit.append(poi)
                else:
                    # 模糊匹配
                    for must_visit_name in must_visit:
                        if (must_visit_name in poi_name) or (poi_name in must_visit_name):
                            found_must_visit.append(poi)
                            break
            
            print(f"必去景点：{len(found_must_visit)}/{len(must_visit)} 个已包含")
            for poi in found_must_visit:
                print(f"  ✓ {poi['name']} (得分: {poi['computed_score']:.3f})")
        
        # 显示前几个候选景点
        if final_candidates:
            print("候选景点（按得分排序）：")
            for i, poi in enumerate(final_candidates[:8]):  # 显示前8个
                print(f"  {i+1}. {poi['name']} (得分: {poi['computed_score']:.3f})")
        
        return final_candidates
        
    except Exception as e:
        print(f"生成候选景点失败: {str(e)}")
        return []

def schedule_pois_across_days(sorted_pois: List[Dict[str, Any]], num_days: int, daily_capacity: int) -> Dict[str, Any]:
    """贪心装箱：按分数排序依次放入每天，不可拆分。返回 daily_plan 与已选列表。"""
    used = set()
    daily_plan: List[List[Dict[str, Any]]] = [[] for _ in range(num_days)]
    remaining = [daily_capacity for _ in range(num_days)]
    for poi in sorted_pois:
        dur = int(poi.get("suggested_duration_hours") or 0)
        if dur <= 0:
            continue
        for day_idx in range(num_days):
            if poi["name"] in used:
                break
            if remaining[day_idx] >= dur:
                daily_plan[day_idx].append(poi)
                remaining[day_idx] -= dur
                used.add(poi["name"])
                break
    selected = [p for p in sorted_pois if p["name"] in used]
    return {"daily_plan": daily_plan, "selected": selected}

def generate_candidate_attractions(structured_info: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：生成满足升级后的时长/预算约束的行程与文本输出。

    规则：
    - 每日可用时长：仅成人10h，否则8h。
    - 每日时长包含景点游玩时长与景点间交通时长（同一日相邻景点）。
    - 预算约束使用总预算；若仅提供每日预算，则按天数折算为总预算。
    - 交通优先选择在预算内“最快”的方式（公交 or 出租车）。
    - 若某候选无法在当前预算/时长下加入，则跳过，继续尝试下一个候选。
    """

    def _parse_cost_to_number(cost: Optional[str]) -> float:
        if cost is None:
            return 0.0
        try:
            # 形如 "25"、"25元"、"25.5元"
            s = str(cost).replace("元", "").strip()
            return float(s) if s else 0.0
        except Exception:
            return 0.0

    def _get_total_budget(budget_obj: Optional[Dict[str, Any]], days: int) -> float:
        if not budget_obj:
            return float("inf")  # 未提供预算则视为无限
        total = budget_obj.get("total")
        per_day = budget_obj.get("per_day")
        if isinstance(total, (int, float)) and total > 0:
            return float(total)
        if isinstance(per_day, (int, float)) and per_day > 0 and days > 0:
            return float(per_day) * float(days)
        return float("inf")

    def _group_size(g: Optional[Dict[str, Any]]) -> int:
        if not g:
            return 1
        return int(g.get("adults") or 0) + int(g.get("children") or 0) + int(g.get("elderly") or 0)

    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def _fallback_route(origin: Dict[str, Any], dest: Dict[str, Any]) -> Dict[str, Any]:
        lo1 = (origin.get("location") or {}).get("lng")
        la1 = (origin.get("location") or {}).get("lat")
        lo2 = (dest.get("location") or {}).get("lng")
        la2 = (dest.get("location") or {}).get("lat")
        if None in (lo1, la1, lo2, la2):
            return {
                "bus_time_min": None,
                "bus_cost": None,
                "taxi_time_min": None,
                "taxi_cost": None,
            }
        dist_km = _haversine_km(float(la1), float(lo1), float(la2), float(lo2))
        taxi_speed_kmh = 30.0
        bus_speed_kmh = 20.0
        taxi_time_min = max(10.0, (dist_km / taxi_speed_kmh) * 60.0)
        bus_time_min = max(15.0, (dist_km / bus_speed_kmh) * 60.0)
        taxi_cost = 13.0 + 2.6 * dist_km
        bus_cost = 2.0 + 0.5 * dist_km
        return {
            "bus_time_min": round(bus_time_min, 1),
            "bus_cost": round(bus_cost, 1),
            "taxi_time_min": round(taxi_time_min, 1),
            "taxi_cost": round(taxi_cost, 1),
        }

    def _route_between(origin: Dict[str, Any], dest: Dict[str, Any]) -> Dict[str, Any]:
        api_key = config.TRANSPORT_API_KEY
        if not api_key:
            return _fallback_route(origin, dest)
        try:
            data = get_route_info(api_key, origin.get("name"), dest.get("name"))
            result = {
                "bus_time_min": data.get("公共交通最短时间"),
                "bus_cost": _parse_cost_to_number(data.get("公共交通费用")),
                "taxi_time_min": data.get("出租车最短时间"),
                "taxi_cost": _parse_cost_to_number(data.get("出租车费用")),
            }
            if result.get("bus_time_min") is None and result.get("taxi_time_min") is None:
                return _fallback_route(origin, dest)
            return result
        except Exception:
            return _fallback_route(origin, dest)

    def _choose_transport_under_budget(route: Dict[str, Any], budget_left: float) -> Optional[Tuple[str, float, float]]:
        """返回 (mode, time_hours, cost_yuan)。优先更快且不超预算；否则选可行较慢；都不行返回None。"""
        options: List[Tuple[str, float, float]] = []
        if route.get("bus_time_min") is not None:
            options.append(("公共交通", float(route["bus_time_min"]) / 60.0, float(route.get("bus_cost") or 0.0)))
        if route.get("taxi_time_min") is not None:
            options.append(("出租车", float(route["taxi_time_min"]) / 60.0, float(route.get("taxi_cost") or 0.0)))
        if not options:
            return None
        # 先按时间升序；在预算内选最快
        options.sort(key=lambda x: x[1])
        feasible = [opt for opt in options if opt[2] <= budget_left]
        if feasible:
            return feasible[0]
        # 若无任何方式满足当前预算余量，则不可行
        return None

    # 加载与预处理
    poi_list = load_poi_data()
    if not poi_list:
        return {"candidates": [], "daily_plan": [], "itinerary_text": ""}

    group = structured_info.get("group") or {}
    preferences = structured_info.get("preferences") or {}
    budget_obj = structured_info.get("budget") or {}
    start_date = structured_info.get("start_date")
    end_date = structured_info.get("end_date")

    daily_capacity = determine_daily_time_budget(group)
    num_days = compute_trip_days(start_date, end_date)
    total_budget = _get_total_budget(budget_obj, num_days)
    people = _group_size(group)

    filtered = [p for p in poi_list if is_poi_suitable_for_group(p, group)]
    scored: List[Dict[str, Any]] = []
    for p in filtered:
        score = compute_poi_score(p, preferences)
        if score <= 0:
            continue
        item = dict(p)
        item["_score"] = round(score, 6)
        scored.append(item)
    scored.sort(key=lambda x: x["_score"], reverse=True)

    # 规划：逐日填充，考虑交通时长与总预算
    daily_plan: List[List[Dict[str, Any]]] = [[] for _ in range(num_days)]
    daily_time_left: List[float] = [float(daily_capacity) for _ in range(num_days)]
    used_names: set[str] = set()
    total_cost: float = 0.0
    legs: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (day_idx, to_item_idx) -> {mode, time_h, cost}

    for day_idx in range(num_days):
        made_progress = True
        while made_progress:
            made_progress = False
            for poi in scored:
                if poi["name"] in used_names:
                    continue
                duration_h = float(poi.get("suggested_duration_hours") or 0.0)
                if duration_h <= 0:
                    continue
                ticket_price = float(poi.get("ticket_price") or 0.0)
                ticket_cost = ticket_price * float(people)

                # 若当天还没有景点，只需要检查该景点时长与票价预算
                if not daily_plan[day_idx]:
                    if duration_h <= daily_time_left[day_idx] and (total_cost + ticket_cost) <= total_budget:
                        daily_plan[day_idx].append(poi)
                        daily_time_left[day_idx] -= duration_h
                        total_cost += ticket_cost
                        used_names.add(poi["name"])
                        made_progress = True
                    continue

                # 若已有景点，需考虑上一景点到该景点的交通
                prev = daily_plan[day_idx][-1]
                route = _route_between(prev, poi)
                # 在总预算余量内，优先选择更快方式
                budget_left_now = total_budget - total_cost - ticket_cost
                choice = _choose_transport_under_budget(route, budget_left_now)
                if not choice:
                    continue
                mode, travel_time_h, travel_cost = choice
                add_time = travel_time_h + duration_h
                add_cost = ticket_cost + travel_cost
                if add_time <= daily_time_left[day_idx] and (total_cost + add_cost) <= total_budget:
                    daily_plan[day_idx].append(poi)
                    daily_time_left[day_idx] -= add_time
                    total_cost += add_cost
                    used_names.add(poi["name"])
                    legs[(day_idx, len(daily_plan[day_idx]) - 1)] = {
                        "mode": mode,
                        "time_h": travel_time_h,
                        "cost": travel_cost,
                    }
                    made_progress = True

    # 轻量候选输出
    candidates = [
        {
            "name": p.get("name"),
            "suggested_duration_hours": p.get("suggested_duration_hours"),
            "popularity_score": p.get("popularity_score"),
            "score": p.get("_score"),
            "tags": p.get("tags"),
            "suitable_for": p.get("suitable_for"),
        }
        for p in scored if p["name"] in used_names
    ]

    # 转为展示计划并组装文本
    display_plan: List[Dict[str, Any]] = []
    lines: List[str] = []
    for idx, day_list in enumerate(daily_plan, start=1):
        display_day = {
            "day": idx,
            "time_budget_hours": daily_capacity,
            "items": [],
        }
        lines.append(f"第{idx}天行程：")
        if not day_list:
            lines.append("（无可行景点安排）")
            display_plan.append(display_day)
            continue
        for j, p in enumerate(day_list):
            dur = float(p.get("suggested_duration_hours") or 0.0)
            ticket = float(p.get("ticket_price") or 0.0) * float(people)
            display_day["items"].append({
                "name": p.get("name"),
                "duration_hours": dur,
                "ticket_cost": ticket,
            })
            lines.append(
                f"景点{j+1}：{p.get('name')}，时长：{int(dur) if dur.is_integer() else round(dur,1)}小时，花费：{int(ticket) if ticket.is_integer() else round(ticket,1)}元"
            )
            # 输出上一段交通（从前一个景点到当前景点），仅当当日景点数≥2
            if j >= 1:
                leg = legs.get((idx - 1, j))
                if leg:
                    t_h = leg["time_h"]
                    c = leg["cost"]
                    lines.insert(-1, f"交通方式：{leg['mode']}，时长：{round(t_h,1)}小时，花费：{int(c) if float(c).is_integer() else round(c,1)}元")
        display_plan.append(display_day)

    return {
        "candidates": candidates,
        "daily_plan": display_plan,
        "total_cost": round(total_cost, 2),
        "itinerary_text": "\n".join(lines),
    }
