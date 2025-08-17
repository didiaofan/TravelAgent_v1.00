import json
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from .models import AgentState, AgentExtraction
from .llm_utils import create_woka_llm, create_parse_prompt, create_parser
from .poi_utils import generate_candidate_attractions

# 必需的顶级字段及其子字段验证
REQUIRED_FIELDS = {
    "departure_city": lambda x: isinstance(x, str) and x.strip() != "",
    "destination_city": lambda x: isinstance(x, str) and x.strip() != "",
    "start_date": lambda x: isinstance(x, str) and len(x) == 10 and x.strip() != "" and x != "2023-10-01" and x != "2023-10-03",
    "end_date": lambda x: isinstance(x, str) and len(x) == 10 and x.strip() != "" and x != "2023-10-01" and x != "2023-10-03",
    "budget": lambda x: isinstance(x, dict) and (("total" in x and x["total"] > 0) or ("per_day" in x and x["per_day"] > 0)),
    "group": lambda x: isinstance(x, dict) and all(k in x for k in ["adults", "children", "elderly"]) and any(v > 0 for k, v in x.items() if k in ["adults", "children", "elderly"]),
    "preferences": lambda x: isinstance(x, dict) and (
        (isinstance(x.get("attraction_types", []), list) and len([i for i in x.get("attraction_types", []) if str(i).strip() != ""]) > 0)
        or (isinstance(x.get("must_visit", []), list) and len([i for i in x.get("must_visit", []) if str(i).strip() != ""]) > 0)
        or (isinstance(x.get("cuisine", []), list) and len([i for i in x.get("cuisine", []) if str(i).strip() != ""]) > 0)
    )
}

# 最大对话轮次限制
MAX_CONVERSATION_STEPS = 10

# 初始化状态
def init_state(user_input: str) -> AgentState:
    return {
        "structured_info": {
            "destination_city": "北京",  # 默认目的地
            "preferences": {"attraction_types": [], "must_visit": [], "cuisine": [], "avoid": [""]},
            "constraints": {"hotel": {"breakfast": True, "family_room": True}, "transport": ""}
        },
        "conversation": [{"role": "user", "content": user_input}],
        "missing_fields": list(REQUIRED_FIELDS.keys()),
        "step_count": 0,
        
        # 约束处理阶段的数据初始化
        "candidate_pois": [],
        "weather_adjusted_pois": [], 
        "daily_time_limit": 12,
        "room_requirements": 1,
        
        # 细粒度的选择结果初始化
        "selected_restaurants": [],
        "selected_hotels": [],
        "transportation_plan": [],
        
        "daily_route_plan": [],
        "time_feasible_routes": [],
        "intensity_feasible_routes": [],
        "budget_feasible_plan": {},
        
        # 约束处理状态初始化
        "constraint_conflicts": [],
        "backtrack_history": [],
        "optimization_attempts": 0
    }

# 解析用户输入节点
def parse_user_input(state: AgentState) -> AgentState:
    # 更新轮次计数器
    state["step_count"] += 1
    
    # 创建解析模板和解析器
    parser = create_parser(AgentExtraction)
    prompt = create_parse_prompt()
    
    # 获取当前结构化信息的JSON字符串
    current_info_str = json.dumps(state["structured_info"], ensure_ascii=False, indent=2)
    
    # 使用沃卡平台的LLM
    llm = create_woka_llm(temperature=0)
    chain = prompt | llm | parser
    
    # 调用LLM解析
    parsed = chain.invoke({
        "current_info": current_info_str,
        "new_input": state["conversation"][-1]["content"],
        "format_instructions": parser.get_format_instructions()
    })
    
    # 兼容不同pydantic版本
    if hasattr(parsed, "model_dump"):
        new_info = parsed.model_dump(exclude_none=True)
    elif hasattr(parsed, "dict"):
        new_info = parsed.dict(exclude_none=True)
    else:
        new_info = dict(parsed)
    
    # 合并新旧信息（新信息覆盖旧信息）
    for key, value in new_info.items():
        if key == "preferences":
            # 合并偏好而不是覆盖
            state["structured_info"].setdefault("preferences", {})
            for pref_key, pref_val in value.items():
                if pref_key in ["attraction_types", "must_visit", "cuisine"]:
                    existing = set(state["structured_info"]["preferences"].get(pref_key, []))
                    new_items = [i for i in pref_val if str(i).strip() != "" and i not in existing]
                    if new_items:
                        state["structured_info"]["preferences"].setdefault(pref_key, []).extend(new_items)
        elif key == "constraints":
            state["structured_info"].setdefault("constraints", {})
            if isinstance(value, dict):
                if "dates" in value:
                    state["structured_info"]["constraints"]["dates"] = value["dates"]
                if "departure_city" in value:
                    state["structured_info"]["constraints"]["departure_city"] = value["departure_city"]
                for constraint_key, constraint_val in value.items():
                    if constraint_key not in ["dates", "departure_city"]:
                        state["structured_info"]["constraints"][constraint_key] = constraint_val
        elif key == "group":
            # 仅接受标准字典结构，由LLM按schema输出；拒绝字符串等无效结构
            if isinstance(value, dict) and all(k in value for k in ["adults", "children", "elderly"]):
                try:
                    state["structured_info"]["group"] = {
                        "adults": int(value.get("adults", 0)),
                        "children": int(value.get("children", 0)),
                        "elderly": int(value.get("elderly", 0)),
                    }
                except Exception:
                    pass
        else:
            # 直接覆盖其他字段
            state["structured_info"][key] = value
    
    return state

# 缺失字段检查节点
def check_missing_fields(state: AgentState) -> AgentState:
    # 重置缺失字段列表
    state["missing_fields"] = []
    
    print(f"\n=== 字段检查调试信息 ===")
    print(f"当前结构化信息: {json.dumps(state['structured_info'], ensure_ascii=False, indent=2)}")
    
    # 检查每个必需字段
    for field, validator in REQUIRED_FIELDS.items():
        print(f"\n检查字段: {field}")
        
        # 字段不存在或验证失败
        if field not in state["structured_info"] or not validator(state["structured_info"][field]):
            # 特殊处理：检查字段是否在其他位置
            if field == "departure_city":
                # 检查是否在根级别
                if "departure_city" in state["structured_info"]:
                    print(f"  ✓ {field} 在根级别找到")
                    continue
                # 检查是否在constraints中
                if "constraints" in state["structured_info"] and "departure_city" in state["structured_info"]["constraints"]:
                    print(f"  ✓ {field} 在constraints中找到")
                    continue
                # 检查是否在travel_info中
                if "travel_info" in state["structured_info"] and "departure_city" in state["structured_info"]["travel_info"]:
                    print(f"  ✓ {field} 在travel_info中找到")
                    continue
                print(f"  ✗ {field} 未找到")
            elif field in ["start_date", "end_date"]:
                # 检查是否在根级别
                if field in state["structured_info"]:
                    # 检查日期是否为空字符串或无效
                    date_value = state["structured_info"][field]
                    if isinstance(date_value, str) and date_value.strip() != "" and len(date_value) == 10:
                        print(f"  ✓ {field} 在根级别找到且有效")
                        continue
                    else:
                        print(f"  ✗ {field} 在根级别找到但无效（空字符串或格式错误）")
                # 检查是否在constraints.dates中
                if "constraints" in state["structured_info"] and "dates" in state["structured_info"]["constraints"]:
                    if field in state["structured_info"]["constraints"]["dates"]:
                        date_value = state["structured_info"]["constraints"]["dates"][field]
                        if isinstance(date_value, str) and date_value.strip() != "" and len(date_value) == 10:
                            print(f"  ✓ {field} 在constraints.dates中找到且有效")
                            continue
                        else:
                            print(f"  ✗ {field} 在constraints.dates中找到但无效")
                # 检查是否在travel_dates中
                if "travel_dates" in state["structured_info"]:
                    if field in state["structured_info"]["travel_dates"]:
                        date_value = state["structured_info"]["travel_dates"][field]
                        if isinstance(date_value, str) and date_value.strip() != "" and len(date_value) == 10:
                            print(f"  ✓ {field} 在travel_dates中找到且有效")
                            continue
                        else:
                            print(f"  ✗ {field} 在travel_dates中找到但无效")
                # 检查是否在dates中
                if "dates" in state["structured_info"]:
                    if field in state["structured_info"]["dates"]:
                        date_value = state["structured_info"]["dates"][field]
                        if isinstance(date_value, str) and date_value.strip() != "" and len(date_value) == 10:
                            print(f"  ✓ {field} 在dates中找到且有效")
                            continue
                        else:
                            print(f"  ✗ {field} 在dates中找到但无效")
                print(f"  ✗ {field} 未找到或无效")
                state["missing_fields"].append(field)
                continue
            elif field == "group":
                # 检查是否在根级别
                if "group" in state["structured_info"]:
                    print(f"  ✓ {field} 在根级别找到")
                    continue
                # 检查是否在budget中
                if "budget" in state["structured_info"] and "group" in state["structured_info"]["budget"]:
                    print(f"  ✓ {field} 在budget中找到")
                    # 将group信息移动到根级别
                    state["structured_info"]["group"] = state["structured_info"]["budget"]["group"]
                    continue
                # 检查是否在其他位置
                for key, value in state["structured_info"].items():
                    if isinstance(value, dict) and "group" in value:
                        print(f"  ✓ {field} 在{key}中找到")
                        # 将group信息移动到根级别
                        state["structured_info"]["group"] = value["group"]
                        break  # 找到后跳出循环
                else:  # 如果没有找到，才添加到缺失字段列表
                    print(f"  ✗ {field} 未找到")
                    state["missing_fields"].append(field)
                    continue
                continue  # 如果找到了，继续下一个字段
            else:
                print(f"  ✗ {field} 未找到或验证失败")
                state["missing_fields"].append(field)
                continue
        
        print(f"  ✓ {field} 验证通过")
        
        # 特殊处理group字段
        if field == "group":
            group = state["structured_info"]["group"]
            if "adults" not in group:
                group["adults"] = 1  # 默认1个成人
    
    print(f"\n最终缺失字段: {state['missing_fields']}")
    return state

# 约束准备节点：派生每天时长与行程天数等规范化约束
def prepare_constraints(state: AgentState) -> AgentState:
    info = state.get("structured_info", {})
    constraints = info.setdefault("constraints", {})
    group = info.get("group") or {}
    start_date = info.get("start_date")
    end_date = info.get("end_date")
    constraints.setdefault("derived", {})
    constraints["derived"]["daily_time_budget_hours"] = determine_daily_time_budget(group)
    constraints["derived"]["trip_days"] = compute_trip_days(start_date, end_date)
    constraints["derived"]["dates"] = {"start_date": start_date, "end_date": end_date}
    return state





# 生成追问节点
def generate_question(state: AgentState) -> AgentState:
    # 检查是否因为天气约束失败需要重新询问日期
    needs_date_change = state.get("needs_date_change", False)
    date_change_reason = state.get("date_change_reason", "")
    
    if needs_date_change:
        content = f"抱歉，根据天气预报分析，{date_change_reason}。\n\n请重新选择您的出行日期，我将为您重新规划行程。请提供新的开始日期和结束日期（格式：YYYY-MM-DD）。"
        state["conversation"].append({
            "role": "assistant",
            "content": content
        })
        # 清除天气约束标记，重置状态以便重新处理
        state["needs_date_change"] = False
        state["date_change_reason"] = ""
        state["weather_constraint_result"] = ""
        # 将日期字段重新标记为缺失，以便重新收集
        state["missing_fields"] = ["start_date", "end_date"]
        return state
    
    if not state["missing_fields"]:
        state["conversation"].append({
            "role": "assistant",
            "content": "信息已收集完整！即将为您生成北京旅行行程。"
        })
        return state

    if state["step_count"] >= MAX_CONVERSATION_STEPS:
        state["conversation"].append({
            "role": "assistant",
            "content": "已达到最大对话轮次，我们将使用当前信息为您规划行程。"
        })
        return state

    missing = set(state["missing_fields"])
    questions: list[str] = []

    # 优先日期
    if "start_date" in missing or "end_date" in missing:
        questions.append("请问您的北京行程开始日期和结束日期分别是什么？格式为YYYY-MM-DD。")
    # 其次人数
    elif "group" in missing:
        questions.append("请问此次同行人数分别是多少？成人、儿童、老人各有几位？")
    # 其次预算
    elif "budget" in missing:
        questions.append("请问此次旅行的预算是多少？可提供总预算或每日预算。")
    # 最后偏好
    elif "preferences" in missing:
        questions.append("请问您对行程有哪些偏好？如景点类型、必去地点、美食偏好或需要避开的项目。")

    if not questions:
        # 兜底：列出缺失字段
        questions.append(f"还有一些信息需要确认：{', '.join(state['missing_fields'])}。请补充一下哦。")

    # 一次只问1条（或未来可扩展为最多2条）
    content = questions[0]

    state["conversation"].append({
        "role": "assistant",
        "content": content
    })
    return state

# 构建LangGraph工作流
def create_agent_workflow():
    workflow = StateGraph(AgentState)
    
    # 添加用户需求收集节点
    workflow.add_node("parse_input", parse_user_input)
    workflow.add_node("check_fields", check_missing_fields)
    workflow.add_node("ask_question", generate_question)
    workflow.add_node("prepare_constraints", prepare_constraints)
    
    # 添加约束处理节点
    workflow.add_node("preference_filter", preference_filter)
    workflow.add_node("team_constraints", team_constraints)
    workflow.add_node("weather_filter", weather_filter)
    
    # 细粒度的路线规划节点
    workflow.add_node("restaurant_selection", restaurant_selection)
    workflow.add_node("hotel_selection", hotel_selection)
    workflow.add_node("transportation_planning", transportation_planning)
    workflow.add_node("route_optimization", route_optimization)
    
    # 检查节点
    workflow.add_node("time_window_check", time_window_check)
    workflow.add_node("intensity_check", intensity_check)
    workflow.add_node("budget_check", budget_check)
    workflow.add_node("conflict_resolution", conflict_resolution)
    
    # 设置入口点
    workflow.set_entry_point("parse_input")
    
    # 用户需求收集阶段的边
    workflow.add_edge("parse_input", "check_fields")
    
    # 条件边 - 决定下一步或结束
    def decide_next_phase(state: AgentState) -> str:
        # 信息完整：进入约束处理阶段；否则继续追问；达到最大轮次直接结束
        if state["step_count"] >= MAX_CONVERSATION_STEPS:
            return END
        if not state["missing_fields"]:
            return "prepare_constraints"
        return "ask_question"
    
    workflow.add_conditional_edges(
        "check_fields",
        decide_next_phase,
        {
            "ask_question": "ask_question",
            "prepare_constraints": "prepare_constraints",
            END: END
        }
    )
    
    # 准备阶段 → 约束处理阶段
    workflow.add_edge("prepare_constraints", "preference_filter")
    
    # 约束处理阶段的边（按照依赖关系）
    workflow.add_edge("preference_filter", "team_constraints")
    workflow.add_edge("team_constraints", "weather_filter") 
    
    # 天气过滤后的条件边：检查是否需要重新选择日期
    def check_weather_constraint_result(state: AgentState) -> str:
        weather_result = state.get("weather_constraint_result", "success")
        needs_date_change = state.get("needs_date_change", False)
        
        if needs_date_change or weather_result in ["extreme_weather_blocking", "must_visit_conflict", "insufficient_fullness"]:
            return END  # 暂时结束，等待用户重新输入日期
        else:
            return "restaurant_selection"
    
    workflow.add_conditional_edges(
        "weather_filter",
        check_weather_constraint_result,
        {
            "restaurant_selection": "restaurant_selection",
            END: END
        }
    )
    workflow.add_edge("restaurant_selection", "hotel_selection")
    workflow.add_edge("hotel_selection", "transportation_planning") 
    workflow.add_edge("transportation_planning", "route_optimization")
    
    # 检查阶段
    workflow.add_edge("route_optimization", "time_window_check")
    workflow.add_edge("time_window_check", "intensity_check")
    workflow.add_edge("intensity_check", "budget_check")
    
    # 预算检查后的条件边：检查是否有约束冲突
    def check_constraints_satisfied(state: AgentState) -> str:
        conflicts = state.get("constraint_conflicts", [])
        if conflicts:
            return "conflict_resolution"
        return END
    
    workflow.add_conditional_edges(
        "budget_check",
        check_constraints_satisfied,
        {
            "conflict_resolution": "conflict_resolution",
            END: END
        }
    )
    
    # 冲突解决后的回退边
    def decide_backtrack_target(state: AgentState) -> str:
        backtrack_history = state.get("backtrack_history", [])
        optimization_attempts = state.get("optimization_attempts", 0)
        
        # 如果尝试次数过多，直接结束
        if optimization_attempts >= 3:
            return END
            
        if backtrack_history:
            return backtrack_history[-1]
        return "preference_filter"  # 默认回退到偏好筛选
    
    workflow.add_conditional_edges(
        "conflict_resolution",
        decide_backtrack_target,
        {
            "preference_filter": "preference_filter",
            "weather_filter": "weather_filter", 
            "restaurant_selection": "restaurant_selection",
            "route_optimization": "route_optimization",
            END: END
        }
    )
    
    # 从追问节点回到解析节点，但需要用户输入（由外层下一轮驱动）
    workflow.add_edge("ask_question", END)
    
    return workflow.compile()

# ==================== 约束处理节点 ====================

# 1. 偏好筛选节点
def preference_filter(state: AgentState) -> AgentState:
    """按景点受欢迎程度和个人偏好生成候选景点列表"""
    from .poi_utils import generate_preference_filtered_candidates
    
    info = state.get("structured_info", {})
    preferences = info.get("preferences", {})
    group = info.get("group", {})
    trip_days = info.get("constraints", {}).get("derived", {}).get("trip_days", 1)
    
    try:
        # 调用专门的候选景点生成函数
        candidates = generate_preference_filtered_candidates(group, preferences, trip_days)
        state["candidate_pois"] = candidates
        
    except Exception as e:
        print(f"偏好筛选节点失败: {str(e)}")
        state["candidate_pois"] = []
    
    return state

# 2. 团队约束节点  
def team_constraints(state: AgentState) -> AgentState:
    """根据团队人数与构成限制游玩时长及住宿配置"""
    info = state.get("structured_info", {})
    group = info.get("group", {})
    
    adults = group.get("adults", 1)
    children = group.get("children", 0) 
    elderly = group.get("elderly", 0)
    
    # 计算每日游玩时间限制
    if elderly > 0 or children > 0:
        daily_time_limit = 9  # 有老人或儿童，每天最多9小时
    else:
        daily_time_limit = 12  # 只有成年人，每天最多12小时
    
    # 计算住宿配置：小孩算0.5个人，总人数求和取整后除以2，商和余数相加
    total_people = adults + (children * 0.5) + elderly
    total_people_rounded = int(total_people)
    quotient = total_people_rounded // 2
    remainder = total_people_rounded % 2
    room_requirements = quotient + remainder
    
    state["daily_time_limit"] = daily_time_limit
    state["room_requirements"] = room_requirements
    
    return state

# 3. 天气过滤节点 - 重新设计的流程
def weather_filter(state: AgentState) -> AgentState:
    """
    根据新的天气约束流程进行筛选
    
    流程：
    A. 检查极端天气是否阻断出行
    B. 检查必去景点是否受天气影响
    C. 筛选完全不可访问的景点
    D. 检查行程饱满度
    """
    import os
    from datetime import datetime, timedelta
    from tools.weather import get_weather_7d
    from .weather_classifier import WeatherClassifier, format_weather_analysis
    
    candidate_pois = state.get("candidate_pois", [])
    info = state.get("structured_info", {})
    
    try:
        # 1. 获取行程日期和团队信息
        start_date = info.get("start_date")
        end_date = info.get("end_date")
        preferences = info.get("preferences", {})
        must_visit_pois = preferences.get("must_visit", [])
        
        if not start_date or not end_date:
            print("⚠️ 缺少行程日期信息，跳过天气过滤")
            state["weather_adjusted_pois"] = candidate_pois
            return state
        
        # 生成行程日期列表
        trip_dates = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_date_obj:
            trip_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
            
        trip_days = len(trip_dates)
        print(f"🗓️ 行程日期: {start_date} 至 {end_date} (共{trip_days}天)")
        
        # 获取团队约束信息
        constraints = info.get("constraints", {}).get("derived", {})
        daily_time_budget = constraints.get("daily_time_budget_hours", 12)
        
        # 2. 获取天气数据
        location_code = "101010100"  # 北京LocationID
        api_host = os.getenv("HEFENG_API_HOST")
        api_key = os.getenv("HEFENG_API_KEY")
        
        if not api_host or not api_key:
            print("⚠️ 缺少天气API配置，跳过天气过滤")
            state["weather_adjusted_pois"] = candidate_pois
            return state
        
        print(f"🌤️ 正在获取北京天气数据...")
        
        response = get_weather_7d(location_code, api_host, api_key)
        
        if response.status_code != 200:
            print(f"❌ 天气API请求失败: {response.status_code}")
            state["weather_adjusted_pois"] = candidate_pois
            return state
        
        weather_data = response.json()
        
        if weather_data.get("code") != "200":
            print(f"❌ 天气API返回错误: {weather_data.get('code')}")
            state["weather_adjusted_pois"] = candidate_pois
            return state
        
        daily_weather = weather_data.get("daily", [])
        print(f"✅ 获取到{len(daily_weather)}天天气数据")
        
        # 3. 分析行程期间天气
        classifier = WeatherClassifier()
        weather_analysis = classifier.analyze_trip_weather(daily_weather, trip_dates)
        
        # 打印天气分析结果
        weather_report = format_weather_analysis(weather_analysis)
        print("\n" + weather_report)
        
        # ================ 新的天气约束流程 ================
        
        print("\n🔍 执行新的天气约束流程...")
        
        # A. 检查是否有极端天气导致不能满足约定的出行天数
        print("\n步骤A: 检查极端天气阻断...")
        is_blocked_by_extreme_weather = classifier.check_extreme_weather_blocking(weather_analysis, trip_days)
        
        if is_blocked_by_extreme_weather:
            print("❌ 极端天气导致无法满足约定出行天数，建议重新选择日期")
            state["weather_constraint_result"] = "extreme_weather_blocking"
            state["weather_adjusted_pois"] = []
            state["weather_analysis"] = weather_analysis
            # 设置需要回到意图输入环节的标记
            state["needs_date_change"] = True
            state["date_change_reason"] = "极端天气导致无法满足约定出行天数"
            return state
        else:
            print("✅ 极端天气检查通过")
            
        # B. 检查必去景点是否受天气影响
        print("\n步骤B: 检查必去景点天气冲突...")
        
        # 获取必去景点的POI信息
        must_visit_poi_objects = []
        if must_visit_pois:
            for must_visit_name in must_visit_pois:
                # 在候选景点中查找必去景点
                for poi in candidate_pois:
                    if must_visit_name in poi.get("name", "") or poi.get("name", "") in must_visit_name:
                        must_visit_poi_objects.append(poi)
                        break
        
        has_must_visit_conflict = classifier.check_must_visit_weather_conflict(weather_analysis, must_visit_poi_objects)
        
        if has_must_visit_conflict:
            print("❌ 必去景点受天气影响无法访问，建议重新选择日期")
            print(f"受影响的必去景点: {[poi.get('name') for poi in must_visit_poi_objects]}")
            state["weather_constraint_result"] = "must_visit_conflict"
            state["weather_adjusted_pois"] = []
            state["weather_analysis"] = weather_analysis
            # 设置需要回到意图输入环节的标记
            state["needs_date_change"] = True
            state["date_change_reason"] = "必去景点受天气影响无法访问"
            return state
        else:
            print("✅ 必去景点天气检查通过")
            
        # C. 筛选完全不可访问的景点
        print("\n步骤C: 筛选完全不可访问的景点...")
        filtered_pois = classifier.filter_completely_inaccessible_pois(candidate_pois, weather_analysis)
        
        print(f"原候选景点: {len(candidate_pois)}个")
        print(f"天气筛选后: {len(filtered_pois)}个")
        
        if len(filtered_pois) < len(candidate_pois):
            removed_count = len(candidate_pois) - len(filtered_pois)
            print(f"因天气原因移除: {removed_count}个景点")
            
            # 显示被移除的景点
            removed_pois = [poi for poi in candidate_pois if poi not in filtered_pois]
            if removed_pois:
                print("被移除的景点:")
                for poi in removed_pois[:5]:  # 只显示前5个
                    indoor_status = poi.get("indoor", "未知")
                    print(f"  - {poi['name']} (室内状态: {indoor_status})")
                if len(removed_pois) > 5:
                    print(f"  ... 还有{len(removed_pois) - 5}个")
        
        # D. 检查行程饱满度
        print("\n步骤D: 检查行程饱满度...")
        is_full, fullness_analysis = classifier.check_trip_fullness(filtered_pois, daily_time_budget, trip_days)
        
        print(f"行程时间预算: {fullness_analysis['total_time_budget']}小时")
        print(f"景点游玩时间: {fullness_analysis['total_suggested_hours']}小时")
        print(f"时间差: {fullness_analysis['time_difference']}小时")
        print(f"饱满度: {fullness_analysis['fullness_percentage']:.1f}%")
        
        if not is_full:
            print("❌ 行程不够饱满，建议重新选择日期")
            state["weather_constraint_result"] = "insufficient_fullness"
            state["weather_adjusted_pois"] = []
            state["weather_analysis"] = weather_analysis
            state["fullness_analysis"] = fullness_analysis
            # 设置需要回到意图输入环节的标记
            state["needs_date_change"] = True
            state["date_change_reason"] = f"行程不够饱满，剩余时间过多({fullness_analysis['time_difference']}小时)"
            return state
        else:
            print("✅ 行程饱满度检查通过")
        
        # E. 成功通过所有检查
        print("\n🎉 天气约束检查全部通过！")
        
        # 显示保留的景点
        if filtered_pois:
            print("\n保留的景点:")
            for poi in filtered_pois[:8]:  # 显示前8个
                indoor_status = poi.get("indoor", "未知")
                duration = poi.get("suggested_duration_hours", 2.0)
                print(f"  ✓ {poi['name']} (室内状态: {indoor_status}, 建议时长: {duration}h)")
            if len(filtered_pois) > 8:
                print(f"  ... 还有{len(filtered_pois) - 8}个")
        
        state["weather_constraint_result"] = "success"
        state["weather_adjusted_pois"] = filtered_pois
        state["weather_analysis"] = weather_analysis
        state["fullness_analysis"] = fullness_analysis
        
    except Exception as e:
        print(f"❌ 天气过滤失败: {str(e)}")
        # 出错时直接传递原候选景点
        state["weather_adjusted_pois"] = candidate_pois
        state["weather_constraint_result"] = "error"
    
    return state

# 4a. 餐厅选择节点
def restaurant_selection(state: AgentState) -> AgentState:
    """根据用户偏好和位置筛选餐厅"""
    weather_adjusted_pois = state.get("weather_adjusted_pois", [])
    info = state.get("structured_info", {})
    preferences = info.get("preferences", {})
    cuisine_prefs = preferences.get("cuisine", [])
    
    # TODO: 实现餐厅筛选逻辑
    # 1. 根据景点位置获取附近餐厅
    # 2. 根据用户偏好筛选餐厅类型
    # 3. 考虑价位和评分
    
    state["selected_restaurants"] = []  # 临时占位
    return state

# 4b. 酒店选择节点  
def hotel_selection(state: AgentState) -> AgentState:
    """根据团队需求和位置选择酒店"""
    info = state.get("structured_info", {})
    room_requirements = state.get("room_requirements", 1)
    start_date = info.get("start_date")
    end_date = info.get("end_date")
    
    # TODO: 集成酒店工具
    # 1. 使用 HotelTool 查询可用酒店
    # 2. 根据房间需求筛选
    # 3. 考虑位置便利性和价格
    
    state["selected_hotels"] = []  # 临时占位
    return state

# 4c. 交通规划节点
def transportation_planning(state: AgentState) -> AgentState:
    """规划各地点间的交通方式和路线"""
    weather_adjusted_pois = state.get("weather_adjusted_pois", [])
    selected_restaurants = state.get("selected_restaurants", [])
    selected_hotels = state.get("selected_hotels", [])
    
    # TODO: 集成路线工具
    # 1. 使用 get_route_info 获取地点间路线信息
    # 2. 比较公共交通vs出租车的时间和费用
    # 3. 根据团队情况选择合适的交通方式
    
    state["transportation_plan"] = []  # 临时占位
    return state

# 4d. 路线优化节点
def route_optimization(state: AgentState) -> AgentState:
    """优化每日路线，确保顺路且满足时间限制"""
    weather_adjusted_pois = state.get("weather_adjusted_pois", [])
    selected_restaurants = state.get("selected_restaurants", [])
    selected_hotels = state.get("selected_hotels", [])
    transportation_plan = state.get("transportation_plan", [])
    daily_time_limit = state.get("daily_time_limit", 12)
    
    # TODO: 实现路线优化算法
    # 1. 按地理位置聚类景点
    # 2. 优化每日游览顺序（最短路径问题）
    # 3. 插入餐厅和休息时间
    # 4. 验证总时间不超过限制
    
    state["daily_route_plan"] = []  # 临时占位
    return state

# 5. 时间窗口检查节点
def time_window_check(state: AgentState) -> AgentState:
    """检查景点开放时间和建议游玩时间是否满足"""
    daily_route_plan = state.get("daily_route_plan", [])
    
    # TODO: 实现时间窗口检查
    # 1. 检查每个景点的开放时间
    # 2. 验证建议游玩时间是否满足
    # 3. 确保到达时间与关闭时间差值 > suggested_duration_hours
    
    state["time_feasible_routes"] = daily_route_plan  # 临时直接传递
    return state

# 6. 强度检查节点
def intensity_check(state: AgentState) -> AgentState:
    """检查每日活动强度是否超过上限"""
    time_feasible_routes = state.get("time_feasible_routes", [])
    
    # TODO: 实现强度检查
    # 1. 计算每日活动强度分值
    # 2. 检查是否超过上限
    # 3. 若超强度，减少当天景点或替换为低强度景点
    
    state["intensity_feasible_routes"] = time_feasible_routes  # 临时直接传递
    return state

# 7. 预算检查节点
def budget_check(state: AgentState) -> AgentState:
    """计算景点、餐饮、酒店、交通总花费，并检查预算"""
    intensity_feasible_routes = state.get("intensity_feasible_routes", [])
    info = state.get("structured_info", {})
    budget = info.get("budget", {})
    room_requirements = state.get("room_requirements", 1)
    
    # TODO: 实现预算检查
    # 1. 计算各项费用
    # 2. 检查是否超预算
    # 3. 若超预算，优先调整高价环节（酒店→餐饮→景点）
    
    state["budget_feasible_plan"] = {
        "routes": intensity_feasible_routes,
        "total_cost": 0,
        "breakdown": {}
    }
    
    # 清空约束冲突（表示所有约束都满足）
    state["constraint_conflicts"] = []
    
    return state

# 8. 冲突解决节点
def conflict_resolution(state: AgentState) -> AgentState:
    """处理约束冲突，决定回退策略"""
    conflicts = state.get("constraint_conflicts", [])
    backtrack_history = state.get("backtrack_history", [])
    optimization_attempts = state.get("optimization_attempts", 0)
    
    # 更新优化尝试次数
    state["optimization_attempts"] = optimization_attempts + 1
    
    # 如果尝试次数过多，使用较宽松的约束
    if optimization_attempts >= 3:
        state["constraint_conflicts"] = []  # 强制结束冲突
        return state
    
    # 根据冲突类型决定回退层级
    if "budget" in conflicts:
        # 预算冲突：回退到路线优化
        backtrack_target = "route_optimization"
    elif "intensity" in conflicts or "time_window" in conflicts:
        # 强度或时间冲突：回退到路线优化  
        backtrack_target = "route_optimization"
    elif "restaurant" in conflicts or "hotel" in conflicts or "transportation" in conflicts:
        # 餐厅/酒店/交通冲突：回退到餐厅选择
        backtrack_target = "restaurant_selection"
    elif "weather" in conflicts:
        # 天气冲突：回退到天气过滤
        backtrack_target = "weather_filter"
    else:
        # 其他冲突：回退到偏好筛选
        backtrack_target = "preference_filter"
    
    # 记录回退历史
    state["backtrack_history"] = backtrack_history + [backtrack_target]
    
    return state

# 从poi_utils导入的函数
def determine_daily_time_budget(group):
    from .poi_utils import determine_daily_time_budget as _determine_daily_time_budget
    return _determine_daily_time_budget(group)

def compute_trip_days(start_date, end_date):
    from .poi_utils import compute_trip_days as _compute_trip_days
    return _compute_trip_days(start_date, end_date)
