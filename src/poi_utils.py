import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    """只有成人：16h；含老人/儿童：12h。"""
    if not group:
        return 12
    num_children = int((group.get("children") or 0))
    num_elderly = int((group.get("elderly") or 0))
    if num_children == 0 and num_elderly == 0:
        return 16
    return 12

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
    bonus = 0.0
    preferred_types = set([t.strip() for t in (preferences.get("attraction_types") or []) if str(t).strip()])
    poi_tags = set([str(t) for t in (poi.get("tags") or [])])
    if preferred_types and (poi_tags & preferred_types):
        bonus += 0.05
    must_visit = set([m.strip() for m in (preferences.get("must_visit") or []) if str(m).strip()])
    if poi.get("name") in must_visit:
        bonus += 0.1
    avoid_list = set([a.strip() for a in (preferences.get("avoid") or []) if str(a).strip()])
    if poi.get("name") in avoid_list or (avoid_list and (poi_tags & avoid_list)):
        return 0.0
    return base + bonus

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
    """主入口：生成满足约束的候选景点与按天计划。"""
    poi_list = load_poi_data()
    if not poi_list:
        return {"candidates": [], "daily_plan": []}
    group = structured_info.get("group") or {}
    preferences = structured_info.get("preferences") or {}
    start_date = structured_info.get("start_date")
    end_date = structured_info.get("end_date")
    daily_capacity = determine_daily_time_budget(group)
    num_days = compute_trip_days(start_date, end_date)
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
    packed = schedule_pois_across_days(scored, num_days=num_days, daily_capacity=daily_capacity)
    candidates = [
        {
            "name": p.get("name"),
            "suggested_duration_hours": p.get("suggested_duration_hours"),
            "popularity_score": p.get("popularity_score"),
            "score": p.get("_score"),
            "tags": p.get("tags"),
            "suitable_for": p.get("suitable_for"),
        }
        for p in packed["selected"]
    ]
    daily_plan_light: List[Dict[str, Any]] = []
    for idx, day_list in enumerate(packed["daily_plan"], start=1):
        daily_plan_light.append({
            "day": idx,
            "time_budget_hours": daily_capacity,
            "items": [
                {
                    "name": p.get("name"),
                    "duration_hours": p.get("suggested_duration_hours"),
                    "score": p.get("_score"),
                }
                for p in day_list
            ]
        })
    return {"candidates": candidates, "daily_plan": daily_plan_light}
