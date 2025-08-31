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
        
        # 新状态图的数据结构初始化
        "daily_candidates": [],
        "selected_restaurants": [],
        "selected_hotels": [],
        "transportation_plan": [],
        
        # 强度相关初始化
        "calculated_intensity": 0.0,
        "intensity_satisfied": True,
        "intensity_optimization_attempts": 0,
        "can_optimize_intensity": False,
        "valid_transport_plans": [],
        
        # 预算相关初始化
        "calculated_cost": 0.0,
        "cost_breakdown": {},
        "budget_satisfied": True,
        "budget_optimization_target": "",
        "recommended_plan": {},
        "all_plan_costs": [],
        "budget_check_result": "",
        
        # 优化控制标记初始化
        "hotel_optimization_blocked": False,
        "transport_optimization_blocked": False,
        "restaurant_optimization_blocked": False,
        "is_optimization_round": False,
        
        # 优化后的数据初始化
        "optimized_hotels": [],
        "optimized_transportation_plan": [],
        "optimized_restaurants": [],
        
        # 每日景点数据初始化
        "daily_available_pois": [],
        
        # 酒店优化相关初始化
        "hotel_optimization_attempts": 0,
        "max_hotel_optimization_attempts": 2,  # 最多优化1次（0=初始，1=第1次优化，2=第2次优化）
        "excluded_hotels": []
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
    
    # 新的节点结构（按照状态图）
    workflow.add_node("scenic_spots_clustering", scenic_spots_clustering)
    workflow.add_node("hotel_selection", hotel_selection)
    workflow.add_node("transportation_planning", transportation_planning)
    workflow.add_node("intensity_calculate", intensity_calculate)
    workflow.add_node("budget_calculate", budget_calculate)
    workflow.add_node("budget_check", budget_check)
    workflow.add_node("final_output", final_output)
    workflow.add_node("hotel_optimization", hotel_optimization)
    
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
            return "scenic_spots_clustering"
    
    workflow.add_conditional_edges(
        "weather_filter",
        check_weather_constraint_result,
        {
            "scenic_spots_clustering": "scenic_spots_clustering",
            END: END
        }
    )
    
    # 按照状态图连接新的节点
    workflow.add_edge("scenic_spots_clustering", "hotel_selection")
    workflow.add_edge("hotel_selection", "transportation_planning")
    workflow.add_edge("transportation_planning", "intensity_calculate")
    
    # intensity_calculate的条件边：根据强度约束检查结果决定下一步
    def decide_after_intensity_check(state: AgentState) -> str:
        intensity_satisfied = state.get("intensity_satisfied", False)
        valid_plans = state.get("valid_transport_plans", [])
        
        print(f"\n🔍 决策检查 - intensity_satisfied: {intensity_satisfied}")
        print(f"🔍 决策检查 - valid_transport_plans数量: {len(valid_plans)}")
        
        if intensity_satisfied:
            print("✅ 强度约束满足，进入预算计算")
            return "budget_calculate"
        else:
            print("❌ 强度约束不满足，流程结束")
            return END
    
    workflow.add_conditional_edges(
        "intensity_calculate",
        decide_after_intensity_check,
        {
            "budget_calculate": "budget_calculate",
            END: END
        }
    )
    
    # budget_calculate 连接到 budget_check
    workflow.add_edge("budget_calculate", "budget_check")
    
    # budget_check的条件边：根据预算检查结果决定下一步
    def decide_after_budget_check(state: AgentState) -> str:
        budget_satisfied = state.get("budget_satisfied", False)
        hotel_optimization_attempts = state.get("hotel_optimization_attempts", 0)
        max_hotel_optimization_attempts = state.get("max_hotel_optimization_attempts", 1)
        
        print(f"\n🔍 预算检查决策 - budget_satisfied: {budget_satisfied}")
        print(f"🔍 预算检查决策 - hotel_optimization_attempts: {hotel_optimization_attempts}")
        print(f"🔍 预算检查决策 - max_hotel_optimization_attempts: {max_hotel_optimization_attempts}")
        
        if budget_satisfied:
            print("✅ 预算满足，输出最终结果")
            return "final_output"
        elif hotel_optimization_attempts < max_hotel_optimization_attempts:
            print(f"⚠️ 预算不满足，尝试优化酒店（第{hotel_optimization_attempts + 1}次，最多{max_hotel_optimization_attempts - 1}次）")
            return "hotel_optimization"  # 直接进入酒店优化，而不是hotel_selection
        else:
            print(f"❌ 预算不满足，已尝试优化{hotel_optimization_attempts}次（最多{max_hotel_optimization_attempts}次），输出现有方案")
            return "final_output"
    
    workflow.add_conditional_edges(
        "budget_check",
        decide_after_budget_check,
        {
            "final_output": "final_output",
            "hotel_optimization": "hotel_optimization"
        }
    )
    
    # 酒店优化后的流程：hotel_selection -> hotel_optimization -> transportation_planning -> intensity_calculate -> budget_calculate -> budget_check
    # 注意：酒店优化后需要重新执行交通规划和强度计算，因为酒店位置变了
    
    # 酒店优化后的流程边（只在预算超限时使用）
    # 正常流程：hotel_selection -> transportation_planning -> intensity_calculate -> budget_calculate -> budget_check
    # 优化流程：hotel_optimization -> transportation_planning -> intensity_calculate -> budget_calculate -> budget_check
    
    # 酒店优化后的流程边
    workflow.add_edge("hotel_optimization", "transportation_planning")
    
    # final_output 结束流程
    workflow.add_edge("final_output", END)

    
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
        daily_time_limit = 9   # 有老人或儿童，每天最多9小时
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

# 3. 天气过滤节点 - 按照新流程设计
def weather_filter(state: AgentState) -> AgentState:
    """
    根据新的天气约束流程进行筛选
    
    新流程：
    A. 检查是否有极端天气导致不能满足约定的出行天数
    B. 检查必去景点是否受天气影响
    C. 根据天气约束情况，生成每日可去景点列表
    D. 检查每天的行程是否饱满
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
        daily_time_budget = state.get("daily_time_limit", 12)
        
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
            
        # C. 根据天气约束情况，生成每日可去景点列表
        print("\n步骤C: 生成每日可去景点列表...")
        daily_available_pois = []
        
        for i, date in enumerate(trip_dates):
            day_weather = weather_analysis.get(date, {})
            
            # 为当天筛选适合的景点
            day_pois = []
            for poi in candidate_pois:
                poi_indoor = poi.get("indoor", "未知")
                
                # 根据天气和景点类型判断是否适合当天访问
                if classifier.is_poi_suitable_for_weather(poi, day_weather):
                    # 为景点添加坐标信息（如果有的话）
                    poi_with_coords = poi.copy()
                    if "coordinates" not in poi_with_coords:
                        # 如果没有坐标信息，可以添加默认坐标或者调用地理编码服务
                        poi_with_coords["coordinates"] = {
                            "latitude": poi.get("lat", 39.9042),  # 北京默认坐标
                            "longitude": poi.get("lon", 116.4074)
                        }
                    
                    day_pois.append(poi_with_coords)
            
            daily_available_pois.append({
                "date": date,
                "weather": day_weather,
                "available_pois": day_pois
            })
            
            print(f"  第{i+1}天 ({date}): {len(day_pois)}个可访问景点")
            
            # 显示部分景点作为示例
            if day_pois:
                for poi in day_pois[:3]:  # 显示前3个
                    indoor_status = poi.get("indoor", "未知")
                    duration = poi.get("suggested_duration_hours", 2.0)
                    score = poi.get("score", 0)
                    print(f"    ✓ {poi['name']} (室内:{indoor_status}, 时长:{duration}h, 得分:{score})")
                if len(day_pois) > 3:
                    print(f"    ... 还有{len(day_pois) - 3}个景点")
        
        # D. 检查每天的行程是否饱满
        print("\n步骤D: 检查每天行程饱满度...")
        all_days_full = True
        insufficient_days = []
        
        for day_info in daily_available_pois:
            date = day_info["date"]
            day_pois = day_info["available_pois"]
            
            # 计算当天所有景点的建议游玩时间总和
            total_suggested_hours = sum(poi.get("suggested_duration_hours", 2.0) for poi in day_pois)
            
            # 计算剩余时间
            remaining_time = daily_time_budget - total_suggested_hours
            
            print(f"  {date}: 可用时间{daily_time_budget}h, 景点总时长{total_suggested_hours}h, 剩余{remaining_time}h")
            
            # 如果剩余时间超过5小时，认为行程不够饱满
            if remaining_time > 5:
                all_days_full = False
                insufficient_days.append(date)
                print(f"    ❌ {date} 行程不够饱满（剩余{remaining_time}小时）")
            else:
                print(f"    ✅ {date} 行程饱满度合适")
        
        if not all_days_full:
            print(f"❌ 行程不够饱满，建议重新选择日期")
            print(f"不够饱满的日期: {', '.join(insufficient_days)}")
            state["weather_constraint_result"] = "insufficient_fullness"
            state["weather_adjusted_pois"] = []
            state["weather_analysis"] = weather_analysis
            state["daily_available_pois"] = daily_available_pois
            # 设置需要回到意图输入环节的标记
            state["needs_date_change"] = True
            state["date_change_reason"] = f"行程不够饱满，以下日期剩余时间过多: {', '.join(insufficient_days)}"
            return state
        else:
            print("✅ 所有日期行程饱满度检查通过")
        
        # E. 成功通过所有检查，生成最终的每日景点列表
        print("\n🎉 天气约束检查全部通过！")
        
        # 将每日可去景点列表扁平化，同时保留每日分组信息
        all_available_pois = []
        for day_info in daily_available_pois:
            for poi in day_info["available_pois"]:
                poi_with_day = poi.copy()
                poi_with_day["available_dates"] = [day_info["date"]]  # 记录该景点可访问的日期
                all_available_pois.append(poi_with_day)
        
        # 合并相同景点的可访问日期
        poi_date_map = {}
        for poi in all_available_pois:
            poi_name = poi["name"]
            if poi_name not in poi_date_map:
                poi_date_map[poi_name] = poi.copy()
            else:
                # 合并可访问日期
                existing_dates = set(poi_date_map[poi_name]["available_dates"])
                new_dates = set(poi["available_dates"])
                poi_date_map[poi_name]["available_dates"] = list(existing_dates.union(new_dates))
        
        final_pois = list(poi_date_map.values())
        
        print(f"\n生成的每日景点列表包含 {len(final_pois)} 个景点")
        for poi in final_pois[:5]:  # 显示前5个
            dates = ', '.join(poi["available_dates"])
            indoor_status = poi.get("indoor", "未知")
            duration = poi.get("suggested_duration_hours", 2.0)
            score = poi.get("score", 0)
            print(f"  ✓ {poi['name']} (可访问日期:{dates}, 室内:{indoor_status}, 时长:{duration}h, 得分:{score})")
        if len(final_pois) > 5:
            print(f"  ... 还有{len(final_pois) - 5}个景点")
        
        state["weather_constraint_result"] = "success"
        state["weather_adjusted_pois"] = final_pois
        state["daily_available_pois"] = daily_available_pois  # 保留每日分组信息
        state["weather_analysis"] = weather_analysis
        
    except Exception as e:
        print(f"❌ 天气过滤失败: {str(e)}")
        # 出错时直接传递原候选景点
        state["weather_adjusted_pois"] = candidate_pois
        state["weather_constraint_result"] = "error"
    
    return state

# ==================== 新的节点函数（按照状态图） ====================

# 1. 景点聚类节点 - scenic_spots_clustering
def scenic_spots_clustering(state: AgentState) -> AgentState:
    """
    智能每日行程分配
    
    改进的多阶段分配策略：
    1. 必去景点优先分配
    2. 高时间消耗景点独立处理
    3. 基于真实地理距离聚类剩余景点
    4. 天气约束优化
    5. 时间预算平衡
    
    核心原则：景点选择一次确定，后续不再调整
    """
    from .improved_clustering import improved_scenic_spots_clustering
    return improved_scenic_spots_clustering(state)

# 2. 酒店选择节点 - hotel_selection
def hotel_selection(state: AgentState) -> AgentState:
    """酒店选择"""
    hotel_optimization_attempts = state.get("hotel_optimization_attempts", 0)
    
    if hotel_optimization_attempts > 0:
        print("🏨 执行酒店优化选择（选择更便宜的酒店）...")
    else:
        print("🏨 执行酒店选择...")
    
    # 获取行程信息
    info = state.get("structured_info", {})
    room_requirements = state.get("room_requirements", 1)
    start_date = info.get("start_date")
    end_date = info.get("end_date")
    group = info.get("group", {})
    
    # 获取已安排的景点信息（用于后续优化酒店位置）
    daily_candidates = state.get("daily_candidates", [])
    
    if not start_date or not end_date:
        print("❌ 缺少入住日期信息，跳过酒店搜索")
        state["selected_hotels"] = []
        return state

    # 转换日期格式（从YYYY-MM-DD转为YYYY/MM/DD）
    checkin_date = start_date.replace('-', '/')
    checkout_date = end_date.replace('-', '/')
    
    # 计算成人和儿童数量
    adults = group.get("adults", 1)
    children = group.get("children", 0)
    
    print(f"🏨 酒店搜索参数:")
    print(f"  目的地: 王府井地铁站")
    print(f"  入住日期: {checkin_date}")
    print(f"  退房日期: {checkout_date}")
    print(f"  房间数: {room_requirements}")
    print(f"  成人数: {adults}")
    print(f"  儿童数: {children}")
    
    # 检查是否为优化模式，如果是则跳过搜索，使用已有结果
    # 初始酒店选择模式：搜索并选择评分最高的酒店
    print("🏨 执行初始酒店选择...")
    
    # 可配置的酒店搜索数量
    max_hotels_config = 5  # 可以从配置文件或参数中读取
    
    try:
        # 调用携程酒店搜索
        from tools.hotel import ctrip_hotel_scraper
        
        print("📡 正在搜索王府井附近酒店...")
        
        hotels_data = ctrip_hotel_scraper(
            destination="王府井",
            checkin=checkin_date,
            checkout=checkout_date,
            rooms=room_requirements,
            adults=adults,
            children=children,
            keyword=None,
            max_hotels=max_hotels_config
        )
        
        # 按评分降序排列酒店（评分高的在前）
        if hotels_data:
            try:
                # 尝试将评分转换为浮点数进行排序
                hotels_data.sort(key=lambda x: float(x['评分']), reverse=True)
                print("✅ 酒店已按评分降序排列")
            except (ValueError, TypeError):
                # 如果评分格式有问题，保持原顺序
                print("⚠️ 评分格式异常，保持原搜索顺序")
        
        # 显示搜索结果统计
        found_count = len(hotels_data)
        
        if found_count < max_hotels_config:
            print(f"\n🏨 搜索到 {found_count} 家酒店 (请求{max_hotels_config}家，实际找到{found_count}家，按评分排序):")
            if found_count > 0:
                print(f"💡 提示: 可能因为时间、价格或房源限制，找到的酒店少于预期")
            else:
                print(f"\n🏨 搜索到 {found_count} 家酒店 (按评分排序):")
        for i, hotel in enumerate(hotels_data, 1):
            print(f"  {i}. {hotel['酒店名称']}")
            print(f"     评分: {hotel['评分']}")
            print(f"     房型: {hotel['房型']}")
            print(f"     价格: {hotel['价格']}")
            print()
        
        # 保存所有搜索结果供后续优化使用（排序后的列表）
        state["hotel_search_results"] = hotels_data
        
        # 初始模式：选择评分最高的酒店
        if hotels_data:
            selected_hotel = hotels_data[0]  # 第一个就是评分最高的
            selection_reason = "评分最高"
            selection_time = "initial"
            
            state["selected_hotels"] = [selected_hotel]
            
            print(f"✅ 选择酒店: {selected_hotel['酒店名称']}")
            print(f"   评分: {selected_hotel['评分']}")
            print(f"   房型: {selected_hotel['房型']}")
            print(f"   价格: {selected_hotel['价格']}")
            print(f"   选择原因: {selection_reason}")
            
            # 添加酒店选择记录
            if "hotel_selection_history" not in state:
                state["hotel_selection_history"] = []
            
            state["hotel_selection_history"].append({
                "selected_hotel": selected_hotel,
                "selection_reason": selection_reason,
                "selection_time": selection_time,
                "available_options": len(hotels_data),
                "max_hotels_requested": max_hotels_config,
                "optimization_attempt": 0  # 初始选择
            })
        else:
            print("❌ 未找到合适的酒店")
            state["selected_hotels"] = []
            state["hotel_selection_history"] = []
        
    except Exception as e:
        print(f"❌ 酒店搜索失败: {str(e)}")
        print("💡 可能的原因:")
        print("  1. 网络连接问题")
        print("  2. Chrome浏览器未在调试模式运行")
        print("  3. 携程网站结构变化")
        print("  4. 搜索参数格式问题")
        
        # 使用备用酒店数据
        fallback_hotel = {
            "酒店名称": "王府井地区酒店（备用）",
            "评分": "4.5",
            "房型": "标准间",
            "价格": "500元/晚"
        }
        
        state["selected_hotels"] = [fallback_hotel]
        state["hotel_search_results"] = [fallback_hotel]
        state["hotel_selection_history"] = [{
            "selected_hotel": fallback_hotel,
            "selection_reason": "搜索失败，使用备用",
            "selection_time": "fallback",
            "available_options": 1,
            "max_hotels_requested": max_hotels_config  # 记录原本请求的数量
        }]
        
        print(f"🔄 使用备用酒店: {fallback_hotel['酒店名称']}")
        print(f"   评分: {fallback_hotel['评分']}")
        print(f"   房型: {fallback_hotel['房型']}")
        print(f"   价格: {fallback_hotel['价格']}")
    
    return state

# 3. 交通规划节点 - transportation_planning  
def transportation_planning(state: AgentState) -> AgentState:
    """
    交通规划节点 - 生成三种优化方案
    
    功能：
    1. 计算每日行程的所有路线（酒店→景点、景点→景点、景点→酒店）
    2. 生成三种交通方案：最省时间、最省金钱、最舒适（全出租车）
    3. 输出详细的路线信息和总计数据
    """
    print("🚗 执行交通规划...")
    
    # 从状态中提取必要数据
    selected_hotels = state.get("selected_hotels", [])
    daily_itinerary = state.get("daily_candidates", [])  # 修正：使用正确的字段名
    
    if not selected_hotels:
        print("❌ 未找到选择的酒店，无法进行交通规划")
        return state
    
    if not daily_itinerary:
        print("❌ 未找到每日行程安排，无法进行交通规划")
        print(f"   调试信息: daily_candidates字段存在吗？{bool(state.get('daily_candidates'))}")
        print(f"   调试信息: 可用的状态字段: {list(state.keys())}")
        return state
    
    # 获取当前选择的酒店名称（确保使用最新的选择）
    hotel_info = selected_hotels[0]
    hotel_name = hotel_info.get("酒店名称", "王府井地区酒店")
    
    # 为高德API添加完整地址格式（市区信息）
    if not hotel_name.startswith("北京"):
        hotel_address = f"北京市东城区{hotel_name}"
    else:
        hotel_address = hotel_name
    
    print(f"🏨 基准酒店: {hotel_name}")
    print(f"🗺️  API地址: {hotel_address}")
    
    # 检查是否是重新计算（酒店有变更）
    previous_transport = state.get("transportation_plans", {})
    if previous_transport and previous_transport.get("hotel_used") != hotel_address:
        print(f"🔄 检测到酒店变更，重新计算交通方案")
        print(f"   之前酒店: {previous_transport.get('hotel_used', '未知')}")
        print(f"   当前酒店: {hotel_address}")
    
    # 验证行程数据结构
    print(f"📋 找到 {len(daily_itinerary)} 天的行程安排")
    for i, day_plan in enumerate(daily_itinerary, 1):
        day_pois = day_plan.get("pois", [])
        day_date = day_plan.get("date", f"第{i}天")
        print(f"   第{i}天 ({day_date}): {len(day_pois)}个景点")
        if day_pois:
            poi_names = [poi.get("name", "未知景点") for poi in day_pois]
            print(f"     景点: {', '.join(poi_names)}")
    
    # 检查是否有高德API密钥
    import os
    
    # 确保加载.env文件
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ 已加载.env环境变量文件")
    except ImportError:
        print("⚠️ 未安装python-dotenv包，请安装: pip install python-dotenv")
    
    api_key = os.getenv("GAODE_API_KEY")  # 修正：使用正确的环境变量名
    if not api_key:
        print("⚠️ 未配置高德API密钥，使用模拟数据进行演示")
        print("   请在.env文件中设置 GAODE_API_KEY=你的高德API密钥")
        return _demo_transportation_planning(state, hotel_address, daily_itinerary)
    else:
        print(f"✅ 已检测到高德API密钥，开始实际路线计算")
        print(f"   API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")  # 部分显示保护隐私
    
    print(f"📊 计算 {len(daily_itinerary)} 天的交通路线...")
    print(f"⏱️  为避免API频率限制，每次请求间隔1秒")
    
    # 计算每日交通路线
    daily_routes = []
    for day_idx, day_plan in enumerate(daily_itinerary, 1):
        day_routes = _calculate_daily_routes(api_key, hotel_address, day_plan, day_idx, hotel_name)
        daily_routes.append(day_routes)
    
    # 生成三种优化方案
    time_optimized = _generate_time_optimized_plan(daily_routes)
    cost_optimized = _generate_cost_optimized_plan(daily_routes)
    comfort_optimized = _generate_comfort_optimized_plan(daily_routes)
    
    # 输出三种方案
    _print_transportation_plans(time_optimized, cost_optimized, comfort_optimized)
    
    # 保存到状态（包含使用的酒店信息）
    state["transportation_plans"] = {
        "time_optimized": time_optimized,
        "cost_optimized": cost_optimized,
        "comfort_optimized": comfort_optimized,
        "daily_routes": daily_routes,
        "hotel_used": hotel_address,  # 记录使用的酒店名称
        "hotel_info": hotel_info      # 完整的酒店信息
    }
    
    print("✅ 交通规划完成")
    return state

def _calculate_daily_routes(api_key: str, hotel_address: str, day_plan: dict, day_idx: int, hotel_name: str = None) -> dict:
    """
    计算单日所有路线的交通信息
    
    Args:
        api_key: 高德API密钥
        hotel_address: 酒店地址
        day_plan: 单日行程计划
        day_idx: 日期索引
        
    Returns:
        dict: 包含所有路线交通信息的数据结构
    """
    from tools.routeinf import get_route_info
    import time
    
    # 如果没有提供hotel_name，从hotel_address中提取
    if hotel_name is None:
        hotel_name = hotel_address.replace("北京市东城区", "").replace("北京市", "")
    
    print(f"\n📅 第{day_idx}天路线计算:")
    
    day_pois = day_plan.get("pois", [])
    if not day_pois:
        print(f"  ⚠️ 第{day_idx}天没有安排景点")
        return {"day": day_idx, "routes": [], "poi_names": []}
    
    poi_names = [poi["name"] for poi in day_pois]
    print(f"  🎯 景点安排: {' → '.join(poi_names)}")
    
    # 为景点地址添加北京市前缀（如果需要）
    def format_address_for_api(address: str) -> str:
        """为高德API格式化地址，添加市区信息"""
        if not address.startswith("北京"):
            return f"北京市{address}"
        return address
    
    routes = []
    
    try:
        # 1. 酒店到第一个景点
        formatted_poi = format_address_for_api(poi_names[0])
        print(f"  🚗 计算路线: {hotel_name} → {poi_names[0]}")
        print(f"     调用API: get_route_info('{hotel_address}', '{formatted_poi}')")
        route_info = get_route_info(api_key, hotel_address, formatted_poi)
        time.sleep(1)  # 延时1秒，避免API请求过频
        routes.append({
            "segment": f"{hotel_name} → {poi_names[0]}",  # 显示用户友好的名称
            "from": hotel_address,
            "to": poi_names[0],
            "route_info": route_info
        })
        
        # 2. 景点之间的路线
        for i in range(len(poi_names) - 1):
            from_poi = poi_names[i]
            to_poi = poi_names[i + 1]
            formatted_from_poi = format_address_for_api(from_poi)
            formatted_to_poi = format_address_for_api(to_poi)
            print(f"  🚗 计算路线: {from_poi} → {to_poi}")
            print(f"     调用API: get_route_info('{formatted_from_poi}', '{formatted_to_poi}')")
            route_info = get_route_info(api_key, formatted_from_poi, formatted_to_poi)
            time.sleep(1)  # 延时1秒，避免API请求过频
            routes.append({
                "segment": f"{from_poi} → {to_poi}",
                "from": from_poi,
                "to": to_poi,
                "route_info": route_info
            })
        
        # 3. 最后一个景点到酒店
        formatted_last_poi = format_address_for_api(poi_names[-1])
        print(f"  🚗 计算路线: {poi_names[-1]} → {hotel_name}")
        print(f"     调用API: get_route_info('{formatted_last_poi}', '{hotel_address}')")
        route_info = get_route_info(api_key, formatted_last_poi, hotel_address)
        time.sleep(1)  # 延时1秒，避免API请求过频
        routes.append({
            "segment": f"{poi_names[-1]} → {hotel_name}",  # 显示用户友好的名称
            "from": poi_names[-1],
            "to": hotel_address,
            "route_info": route_info
        })
        
        print(f"  ✅ 第{day_idx}天共计算 {len(routes)} 条路线")
        
    except Exception as e:
        print(f"  ❌ 第{day_idx}天路线计算失败: {str(e)}")
        print(f"  💡 可能原因:")
        print(f"     - API请求频率过高，建议增加延时")
        print(f"     - 地址名称无法识别或编码失败")
        print(f"     - 网络连接问题或API服务暂不可用")
        print(f"     - API密钥配额不足或权限问题")
        print(f"  🔄 使用模拟数据继续计算")
        routes = _generate_mock_routes(hotel_address, poi_names, day_idx)
    
    return {
        "day": day_idx,
        "routes": routes,
        "poi_names": poi_names,
        "date": day_plan.get("date", f"第{day_idx}天")
    }

def _generate_mock_routes(hotel_address: str, poi_names: list, day_idx: int) -> list:
    """生成模拟路线数据（用于API调用失败时）"""
    import random
    
    routes = []
    
    # 提取酒店名称（去除北京市东城区前缀）
    hotel_name = hotel_address.replace("北京市东城区", "").replace("北京市", "")
    
    # 酒店到第一个景点
    routes.append({
        "segment": f"{hotel_name} → {poi_names[0]}",  # 显示用户友好的名称
        "from": hotel_address,
        "to": poi_names[0],
        "route_info": {
            "出发地": hotel_address,
            "目的地": poi_names[0],
            "公共交通最短时间": round(random.uniform(20, 45), 1),
            "公共交通费用": f"{random.randint(3, 8)}元",
            "出租车最短时间": round(random.uniform(15, 35), 1),
            "出租车费用": f"{random.randint(25, 60)}元"
        }
    })
    
    # 景点之间的路线
    for i in range(len(poi_names) - 1):
        routes.append({
            "segment": f"{poi_names[i]} → {poi_names[i+1]}",
            "from": poi_names[i],
            "to": poi_names[i+1],
            "route_info": {
                "出发地": poi_names[i],
                "目的地": poi_names[i+1],
                "公共交通最短时间": round(random.uniform(15, 40), 1),
                "公共交通费用": f"{random.randint(2, 6)}元",
                "出租车最短时间": round(random.uniform(10, 30), 1),
                "出租车费用": f"{random.randint(20, 50)}元"
            }
        })
    
    # 最后一个景点到酒店
    routes.append({
        "segment": f"{poi_names[-1]} → {hotel_name}",  # 显示用户友好的名称
        "from": poi_names[-1],
        "to": hotel_address,
        "route_info": {
            "出发地": poi_names[-1],
            "目的地": hotel_address,
            "公共交通最短时间": round(random.uniform(20, 45), 1),
            "公共交通费用": f"{random.randint(3, 8)}元",
            "出租车最短时间": round(random.uniform(15, 35), 1),
            "出租车费用": f"{random.randint(25, 60)}元"
        }
    })
    
    return routes

def _generate_time_optimized_plan(daily_routes: list) -> dict:
    """生成最省时间的交通方案"""
    plan = {
        "strategy": "最省时间",
        "description": "每条路线选择耗时最短的交通方式",
        "daily_plans": [],
        "total_time": 0,
        "total_cost": 0
    }
    
    for day_routes in daily_routes:
        day_plan = {
            "day": day_routes["day"],
            "date": day_routes["date"],
            "routes": [],
            "day_total_time": 0,
            "day_total_cost": 0
        }
        
        for route in day_routes["routes"]:
            route_info = route["route_info"]
            
            # 选择时间最短的方式
            bus_time = route_info.get("公共交通最短时间", float('inf'))
            taxi_time = route_info.get("出租车最短时间", float('inf'))
            
            if bus_time <= taxi_time:
                selected_method = "公共交通"
                selected_time = bus_time
                selected_cost = route_info.get("公共交通费用", "0元")
            else:
                selected_method = "出租车"
                selected_time = taxi_time
                selected_cost = route_info.get("出租车费用", "0元")
            
            # 提取费用数字
            cost_num = float(''.join(filter(str.isdigit, selected_cost.replace('元', ''))))
            
            route_plan = {
                "segment": route["segment"],
                "method": selected_method,
                "time": selected_time,
                "cost": selected_cost,
                "cost_num": cost_num
            }
            
            day_plan["routes"].append(route_plan)
            day_plan["day_total_time"] += selected_time
            day_plan["day_total_cost"] += cost_num
        
        plan["daily_plans"].append(day_plan)
        plan["total_time"] += day_plan["day_total_time"]
        plan["total_cost"] += day_plan["day_total_cost"]
    
    return plan

def _generate_cost_optimized_plan(daily_routes: list) -> dict:
    """生成最省金钱的交通方案"""
    plan = {
        "strategy": "最省金钱",
        "description": "每条路线选择费用最低的交通方式",
        "daily_plans": [],
        "total_time": 0,
        "total_cost": 0
    }
    
    for day_routes in daily_routes:
        day_plan = {
            "day": day_routes["day"],
            "date": day_routes["date"],
            "routes": [],
            "day_total_time": 0,
            "day_total_cost": 0
        }
        
        for route in day_routes["routes"]:
            route_info = route["route_info"]
            
            # 提取费用数字进行比较
            bus_cost_str = route_info.get("公共交通费用", "999元")
            taxi_cost_str = route_info.get("出租车费用", "999元")
            
            bus_cost = float(''.join(filter(str.isdigit, bus_cost_str.replace('元', ''))))
            taxi_cost = float(''.join(filter(str.isdigit, taxi_cost_str.replace('元', ''))))
            
            # 选择费用最低的方式
            if bus_cost <= taxi_cost:
                selected_method = "公共交通"
                selected_time = route_info.get("公共交通最短时间", 0)
                selected_cost = bus_cost_str
                cost_num = bus_cost
            else:
                selected_method = "出租车"
                selected_time = route_info.get("出租车最短时间", 0)
                selected_cost = taxi_cost_str
                cost_num = taxi_cost
            
            route_plan = {
                "segment": route["segment"],
                "method": selected_method,
                "time": selected_time,
                "cost": selected_cost,
                "cost_num": cost_num
            }
            
            day_plan["routes"].append(route_plan)
            day_plan["day_total_time"] += selected_time
            day_plan["day_total_cost"] += cost_num
        
        plan["daily_plans"].append(day_plan)
        plan["total_time"] += day_plan["day_total_time"]
        plan["total_cost"] += day_plan["day_total_cost"]
    
    return plan

def _generate_comfort_optimized_plan(daily_routes: list) -> dict:
    """生成最舒适的交通方案（全出租车）"""
    plan = {
        "strategy": "最舒适",
        "description": "全程使用出租车，提供最佳舒适度",
        "daily_plans": [],
        "total_time": 0,
        "total_cost": 0
    }
    
    for day_routes in daily_routes:
        day_plan = {
            "day": day_routes["day"],
            "date": day_routes["date"],
            "routes": [],
            "day_total_time": 0,
            "day_total_cost": 0
        }
        
        for route in day_routes["routes"]:
            route_info = route["route_info"]
            
            # 全部使用出租车
            selected_method = "出租车"
            selected_time = route_info.get("出租车最短时间", 0)
            selected_cost = route_info.get("出租车费用", "0元")
            
            # 提取费用数字
            cost_num = float(''.join(filter(str.isdigit, selected_cost.replace('元', ''))))
            
            route_plan = {
                "segment": route["segment"],
                "method": selected_method,
                "time": selected_time,
                "cost": selected_cost,
                "cost_num": cost_num
            }
            
            day_plan["routes"].append(route_plan)
            day_plan["day_total_time"] += selected_time
            day_plan["day_total_cost"] += cost_num
        
        plan["daily_plans"].append(day_plan)
        plan["total_time"] += day_plan["day_total_time"]
        plan["total_cost"] += day_plan["day_total_cost"]
    
    return plan

def _print_transportation_plans(time_plan: dict, cost_plan: dict, comfort_plan: dict):
    """格式化输出三种交通方案"""
    
    print("\n" + "="*80)
    print("🚗 交通规划方案对比")
    print("="*80)
    
    plans = [time_plan, cost_plan, comfort_plan]
    
    for plan in plans:
        print(f"\n📋 【{plan['strategy']}方案】")
        print(f"   策略说明: {plan['description']}")
        print(f"   总出行时长: {plan['total_time']:.1f}分钟")
        print(f"   总出行费用: {plan['total_cost']:.0f}元")
        print("-" * 60)
        
        for day_plan in plan["daily_plans"]:
            print(f"\n📅 {day_plan['date']} (第{day_plan['day']}天)")
            print(f"   当日交通时长: {day_plan['day_total_time']:.1f}分钟")
            print(f"   当日交通费用: {day_plan['day_total_cost']:.0f}元")
            
            for i, route in enumerate(day_plan["routes"], 1):
                print(f"   {i}. {route['segment']}")
                print(f"      交通方式: {route['method']}")
                print(f"      耗时: {route['time']:.1f}分钟")
                print(f"      费用: {route['cost']}")
        
        print("-" * 60)
    
    # 方案对比表
    print(f"\n📊 方案对比总表:")
    print(f"{'方案类型':<12} {'总时长(分钟)':<12} {'总费用(元)':<12} {'特点'}")
    print("-" * 50)
    
    for plan in plans:
        features = {
            "最省时间": "时间最短",
            "最省金钱": "费用最低", 
            "最舒适": "全程出租车"
        }
        feature = features.get(plan['strategy'], "")
        print(f"{plan['strategy']:<12} {plan['total_time']:<12.1f} {plan['total_cost']:<12.0f} {feature}")
    
    print("="*80)

def _demo_transportation_planning(state: dict, hotel_address: str, daily_itinerary: list) -> dict:
    """演示模式的交通规划（无API时使用）"""
    print("🎭 演示模式：生成模拟交通数据")
    
    # 生成模拟的每日路线
    daily_routes = []
    for day_idx, day_plan in enumerate(daily_itinerary, 1):
        day_pois = day_plan.get("pois", [])
        if day_pois:
            poi_names = [poi["name"] for poi in day_pois]
            routes = _generate_mock_routes(hotel_address, poi_names, day_idx)
            daily_routes.append({
                "day": day_idx,
                "routes": routes,
                "poi_names": poi_names,
                "date": day_plan.get("date", f"第{day_idx}天")
            })
    
    # 生成三种方案
    time_optimized = _generate_time_optimized_plan(daily_routes)
    cost_optimized = _generate_cost_optimized_plan(daily_routes)
    comfort_optimized = _generate_comfort_optimized_plan(daily_routes)
    
    # 输出方案
    _print_transportation_plans(time_optimized, cost_optimized, comfort_optimized)
    
    # 保存到状态
    state["transportation_plans"] = {
        "time_optimized": time_optimized,
        "cost_optimized": cost_optimized,
        "comfort_optimized": comfort_optimized,
        "daily_routes": daily_routes,
        "demo_mode": True,
        "hotel_used": hotel_address,  # 记录使用的酒店名称
        "hotel_info": state.get("selected_hotels", [{}])[0]  # 完整的酒店信息
    }
    
    print("✅ 演示交通规划完成")
    return state

# 4. 强度计算节点 - intensity_calculate
def intensity_calculate(state: AgentState) -> AgentState:
    """
    强度计算节点 - 计算不同交通方式的每日行程耗时并检查约束
    
    功能：
    1. 计算每日总耗时（景点游玩时间 + 交通时间）
    2. 显示三种交通方案的详细时间分解
    3. 以小时为计算单位
    4. 检查强度是否满足team_constraints约束
    5. 如果有满足约束的方案，保存到state并进入budget_calculate
    6. 如果没有满足约束的方案，直接结束流程
    """
    print("💪 执行强度计算和约束检查...")
    
    # 提取必要数据
    daily_candidates = state.get("daily_candidates", [])
    transportation_plans = state.get("transportation_plans", {})
    daily_time_limit = state.get("daily_time_limit", 12)  # 从team_constraints获取每日时间限制
    
    if not daily_candidates:
        print("❌ 未找到每日行程安排，无法进行强度计算")
        state["intensity_satisfied"] = False
        return state

    if not transportation_plans:
        print("❌ 未找到交通规划方案，无法进行强度计算")
        state["intensity_satisfied"] = False
        return state
    
    print(f"📊 计算三种交通方案的每日行程耗时...")
    print(f"⏰ 每日时间约束: {daily_time_limit}小时")
    
    # 计算并显示三种交通方案的强度
    intensity_results = {}
    
    for plan_name, plan_data in transportation_plans.items():
        if plan_name in ["time_optimized", "cost_optimized", "comfort_optimized"]:
            result = _calculate_plan_intensity_simple(daily_candidates, plan_data)
            intensity_results[plan_name] = result
            _print_intensity_simple(plan_name, result)
    
    # 保存计算结果
    state["intensity_calculation_result"] = intensity_results
    
    # === 强度约束检查 ===
    print(f"\n🔍 开始强度约束检查...")
    
    valid_plans = []
    invalid_plans = []
    
    for plan_name, plan_data in intensity_results.items():
        strategy = plan_data.get("strategy", plan_name)
        daily_details = plan_data.get("daily_details", [])
        
        print(f"\n📋 检查【{strategy}】方案:")
        
        # 检查每日是否超过时间限制
        is_plan_valid = True
        exceeded_days = []
        
        for day_detail in daily_details:
            date = day_detail.get("date", "")
            total_day_hours = day_detail.get("total_hours", 0)
            
            if total_day_hours > daily_time_limit:
                is_plan_valid = False
                exceed_hours = total_day_hours - daily_time_limit
                exceeded_days.append({
                    "date": date,
                    "total_hours": total_day_hours,
                    "exceed_hours": exceed_hours
                })
                print(f"  ❌ {date}: {total_day_hours:.1f}h > {daily_time_limit}h (超出{exceed_hours:.1f}h)")
            else:
                print(f"  ✅ {date}: {total_day_hours:.1f}h ≤ {daily_time_limit}h")
        
        # 构建方案信息
        plan_info = {
            "plan_name": plan_name,
            "strategy": strategy,
            "is_valid": is_plan_valid,
            "total_hours": plan_data.get("total_hours", 0),
            "avg_daily_hours": plan_data.get("avg_daily_hours", 0),
            "exceeded_days": exceeded_days,
            "daily_details": daily_details
        }
        
        if is_plan_valid:
            print(f"  ✅ 【{strategy}】方案符合强度约束")
            valid_plans.append(plan_info)
        else:
            print(f"  ❌ 【{strategy}】方案超出强度约束 (超限天数: {len(exceeded_days)})")
            invalid_plans.append(plan_info)
    
    # 检查结果摘要
    print(f"\n📊 强度约束检查结果:")
    print(f"  符合约束的方案: {len(valid_plans)}个")
    print(f"  不符合约束的方案: {len(invalid_plans)}个")
    
    if valid_plans:
        print(f"\n✅ 可行方案列表:")
        for plan in valid_plans:
            print(f"    - {plan['strategy']}: 平均每日{plan['avg_daily_hours']:.1f}h")
        
        # 保存有效方案到state
        state["valid_transport_plans"] = valid_plans
        state["intensity_satisfied"] = True
        
        print(f"\n🎯 {len(valid_plans)}个符合约束的方案已保存到state")
        print("🔄 准备进入预算计算节点...")
        
    else:
        print(f"\n❌ 没有方案满足强度约束")
        if invalid_plans:
            print(f"不可行方案:")
            for plan in invalid_plans:
                print(f"    - {plan['strategy']}: 平均每日{plan['avg_daily_hours']:.1f}h (超限{len(plan['exceeded_days'])}天)")
        
        state["valid_transport_plans"] = []
        state["intensity_satisfied"] = False
        
        print("\n" + "="*60)
        print("❌ 强度约束检查失败")
        print("="*60)
        print(f"所有交通方案都超出了每日{daily_time_limit}小时的时间限制。")
        print("建议:")
        print("1. 减少每日景点数量")
        print("2. 选择游玩时间更短的景点")
        print("3. 调整团队约束（如增加每日游玩时间限制）")
        print("="*60)
        print("🛑 流程结束")
    
    print("✅ 强度计算和约束检查完成")
    print(f"🔍 函数结束时 - intensity_satisfied: {state.get('intensity_satisfied', '未设置')}")
    print(f"🔍 函数结束时 - valid_transport_plans数量: {len(state.get('valid_transport_plans', []))}")
    return state

# 5. 预算计算节点 - budget_calculate
def budget_calculate(state: AgentState) -> AgentState:
    """
    预算计算节点 - 计算总旅行费用
    
    功能：
    1. 计算景点门票费用（门票价格 * 人数）
    2. 计算酒店费用（游玩天数 * 房间数）
    3. 计算交通费用（符合约束的交通方式费用）
    4. 求和并输出总费用
    5. 确认符合预算的最优方案
    """
    print("💰 执行预算计算...")
    
    # 获取基础数据
    valid_plans = state.get("valid_transport_plans", [])
    daily_candidates = state.get("daily_candidates", [])
    selected_hotels = state.get("selected_hotels", [])
    info = state.get("structured_info", {})
    
    if not valid_plans:
        print("❌ 没有符合强度约束的交通方案，无法进行预算计算")
        return state
    
    if not daily_candidates:
        print("❌ 没有每日行程数据，无法计算景点门票费用")
        return state
    
    if not selected_hotels:
        print("❌ 没有选择的酒店，无法计算住宿费用")
        return state
    
    # 获取基本信息
    group = info.get("group", {})
    total_people = group.get("adults", 1) + group.get("children", 0) + group.get("elderly", 0)
    room_requirements = state.get("room_requirements", 1)
    trip_days = len(daily_candidates)
    budget_info = info.get("budget", {})
    
    print(f"📊 预算计算参数:")
    print(f"  总人数: {total_people}人")
    print(f"  房间数: {room_requirements}间")
    print(f"  行程天数: {trip_days}天")
    
    # 1. 计算景点门票费用
    print(f"\n🎫 计算景点门票费用...")
    total_ticket_cost = 0
    ticket_details = []
    
    for day_info in daily_candidates:
        date = day_info.get("date", "")
        pois = day_info.get("pois", [])
        
        day_ticket_cost = 0
        day_tickets = []
        
        for poi in pois:
            poi_name = poi.get("name", "")
            ticket_price = _get_poi_ticket_price(poi)
            poi_ticket_cost = ticket_price * total_people
            
            day_ticket_cost += poi_ticket_cost
            day_tickets.append({
                "poi_name": poi_name,
                "ticket_price": ticket_price,
                "people_count": total_people,
                "total_cost": poi_ticket_cost
            })
            
            print(f"    {poi_name}: {ticket_price}元/人 × {total_people}人 = {poi_ticket_cost}元")
        
        total_ticket_cost += day_ticket_cost
        ticket_details.append({
            "date": date,
            "day_cost": day_ticket_cost,
            "tickets": day_tickets
        })
        
        print(f"  {date} 门票小计: {day_ticket_cost}元")
    
    print(f"🎫 景点门票总费用: {total_ticket_cost}元")
    
    # 2. 计算酒店费用
    print(f"\n🏨 计算酒店费用...")
    hotel_info = selected_hotels[0]
    hotel_name = hotel_info.get("酒店名称", "")
    hotel_price_str = hotel_info.get("价格", "500元/晚")
    
    # 提取酒店价格数字
    try:
        hotel_price_per_night = float(''.join(filter(str.isdigit, hotel_price_str)))
    except:
        hotel_price_per_night = 500  # 默认价格
    
    total_hotel_cost = hotel_price_per_night * room_requirements * trip_days
    
    print(f"  酒店: {hotel_name}")
    print(f"  价格: {hotel_price_per_night}元/晚/间 × {room_requirements}间 × {trip_days}天 = {total_hotel_cost}元")
    print(f"🏨 酒店总费用: {total_hotel_cost}元")
    
    # 3. 计算各个交通方案的费用并选择最优方案
    print(f"\n🚗 计算交通费用并选择最优方案...")
    
    plan_costs = []
    budget_limit = budget_info.get("total") or (budget_info.get("per_day", 1000) * trip_days)
    
    for plan in valid_plans:
        strategy = plan.get("strategy", "")
        plan_name = plan.get("plan_name", "")
        
        # 重新计算交通费用，考虑公共交通的人数问题
        transport_cost = _calculate_transport_cost_with_people(
            state.get("transportation_plans", {}).get(plan_name, {}), 
            total_people
        )
        
        # 计算总费用
        total_cost = total_ticket_cost + total_hotel_cost + transport_cost
        
        plan_cost_info = {
            "strategy": strategy,
            "plan_name": plan_name,
            "ticket_cost": total_ticket_cost,
            "hotel_cost": total_hotel_cost,
            "transport_cost": transport_cost,
            "total_cost": total_cost,
            "within_budget": total_cost <= budget_limit,
            "budget_difference": total_cost - budget_limit
        }
        
        plan_costs.append(plan_cost_info)
        
        print(f"\n📋 【{strategy}】方案费用明细:")
        print(f"    🎫 景点门票: {total_ticket_cost}元")
        print(f"    🏨 酒店住宿: {total_hotel_cost}元")
        print(f"    🚗 交通费用: {transport_cost}元")
        print(f"    💰 总费用: {total_cost}元")
        
        if budget_limit > 0:
            if total_cost <= budget_limit:
                print(f"    预算状态: ✅ 符合预算 (预算{budget_limit}元，剩余{budget_limit - total_cost}元)")
            else:
                print(f"    预算状态: ❌ 超出预算 (预算{budget_limit}元，超出{total_cost - budget_limit}元)")
    
    # 选择最优方案
    print(f"\n🎯 选择最优方案...")
    
    # 优先选择符合预算的方案中费用最低的
    within_budget_plans = [p for p in plan_costs if p["within_budget"]]
    
    if within_budget_plans:
        # 在符合预算的方案中选择费用最低的
        best_plan = min(within_budget_plans, key=lambda x: x["total_cost"])
        print(f"✅ 选择符合预算的最优方案: 【{best_plan['strategy']}】")
    else:
        # 如果都超预算，选择超出最少的
        best_plan = min(plan_costs, key=lambda x: x["budget_difference"])
        print(f"⚠️ 所有方案都超预算，选择超出最少的方案: 【{best_plan['strategy']}】")
    
    print(f"\n💰 最终推荐方案:")
    print(f"  方案: {best_plan['strategy']}")
    print(f"  景点门票: {best_plan['ticket_cost']}元")
    print(f"  酒店住宿: {best_plan['hotel_cost']}元") 
    print(f"  交通费用: {best_plan['transport_cost']}元")
    print(f"  总费用: {best_plan['total_cost']}元")
    
    if budget_limit > 0:
        if best_plan["within_budget"]:
            print(f"  预算状态: ✅ 符合预算 (预算{budget_limit}元)")
        else:
            print(f"  预算状态: ⚠️ 超出预算 {best_plan['budget_difference']}元")
    
    # 4. 汇总所有可行方案的花费
    print(f"\n📊 所有可行方案费用汇总:")
    print("=" * 80)
    print(f"{'方案类型':<15} {'景点门票':<10} {'酒店住宿':<10} {'交通费用':<10} {'总费用':<10} {'预算状态'}")
    print("-" * 80)
    
    for plan_cost in plan_costs:
        status = "✅符合" if plan_cost["within_budget"] else "❌超出"
        print(f"{plan_cost['strategy']:<15} {plan_cost['ticket_cost']:<10} {plan_cost['hotel_cost']:<10} {plan_cost['transport_cost']:<10} {plan_cost['total_cost']:<10} {status}")
    
    print("=" * 80)
    
    # 保存结果到state
    state["calculated_cost"] = best_plan["total_cost"]
    state["cost_breakdown"] = {
        "ticket_cost": best_plan["ticket_cost"],
        "hotel_cost": best_plan["hotel_cost"], 
        "transport_cost": best_plan["transport_cost"],
        "ticket_details": ticket_details,  # 保存门票详情
        "hotel_details": {
            "hotel_name": hotel_name,
            "price_per_night": hotel_price_per_night,
            "rooms": room_requirements,
            "nights": trip_days
        }
    }
    state["budget_satisfied"] = best_plan["within_budget"]
    state["recommended_plan"] = best_plan
    state["all_plan_costs"] = plan_costs
    
    print(f"\n🎉 预算计算完成！")
    print(f"✅ 已确认最优旅行方案: 【{best_plan['strategy']}】")
    print(f"💰 总费用: {best_plan['total_cost']}元")
    if budget_limit > 0:
        if best_plan["within_budget"]:
            print(f"📊 预算状态: 符合预算，剩余 {budget_limit - best_plan['total_cost']}元")
        else:
            print(f"📊 预算状态: 超出预算 {best_plan['budget_difference']}元")
    
    return state

def _get_poi_ticket_price(poi: dict) -> float:
    """获取POI的门票价格"""
    # 尝试从POI数据中获取门票价格
    if "ticket_price" in poi:
        price = poi["ticket_price"]
        if isinstance(price, (int, float)):
            return float(price)
        elif isinstance(price, str):
            # 提取字符串中的数字
            try:
                return float(''.join(filter(str.isdigit, price)))
            except:
                pass
    
    # 如果没有门票价格信息，使用默认价格
    poi_name = poi.get("name", "")
    return _get_default_ticket_price(poi_name)

def _calculate_transport_cost_with_people(transport_plan: dict, total_people: int) -> float:
    """
    重新计算交通费用，考虑公共交通的人数问题
    
    Args:
        transport_plan: 交通方案数据
        total_people: 总人数
    
    Returns:
        float: 考虑人数后的总交通费用
    """
    if not transport_plan:
        return 0
    
    strategy = transport_plan.get("strategy", "")
    daily_plans = transport_plan.get("daily_plans", [])
    
    print(f"\n🚗 计算【{strategy}】方案交通费用 (人数: {total_people}人):")
    
    total_transport_cost = 0
    
    for day_plan in daily_plans:
        date = day_plan.get("date", "")
        routes = day_plan.get("routes", [])
        day_cost = 0
        
        print(f"  📅 {date}:")
        
        for route in routes:
            segment = route.get("segment", "")
            method = route.get("method", "")
            cost_str = route.get("cost", "0元")
            
            # 提取费用数字
            cost_per_person = float(''.join(filter(str.isdigit, cost_str.replace('元', ''))))
            
            # 如果是公共交通，需要乘以人数；如果是出租车，不需要
            if method == "公共交通":
                route_total_cost = cost_per_person * total_people
                print(f"    {segment}: {method} {cost_per_person}元/人 × {total_people}人 = {route_total_cost}元")
            else:  # 出租车
                route_total_cost = cost_per_person
                print(f"    {segment}: {method} {cost_per_person}元 (全车价格)")
            
            day_cost += route_total_cost
        
        print(f"  📅 {date} 交通小计: {day_cost}元")
        total_transport_cost += day_cost
    
    print(f"🚗 【{strategy}】交通总费用: {total_transport_cost}元")
    return total_transport_cost

def _select_cheaper_hotel(state: AgentState, hotels_data: list) -> dict:
    """
    从候选酒店中选择更便宜的酒店
    
    Args:
        state: 当前状态，包含之前选择的酒店信息
        hotels_data: 酒店搜索结果列表
    
    Returns:
        dict: 选择的更便宜的酒店
    """
    # 获取当前选择的酒店价格
    current_hotels = state.get("selected_hotels", [])
    if not current_hotels:
        # 如果没有当前酒店，选择最便宜的
        return min(hotels_data, key=lambda x: _extract_hotel_price(x.get("价格", "999元")))
    
    current_hotel = current_hotels[0]
    current_hotel_name = current_hotel.get("酒店名称", "")
    current_price = _extract_hotel_price(current_hotel.get("价格", "999元"))
    
    # 获取排除列表
    excluded_hotels = state.get("excluded_hotels", [])
    
    print(f"🔍 当前酒店: {current_hotel_name}")
    print(f"🔍 当前酒店价格: {current_price}元/晚")
    print(f"🔍 排除列表: {excluded_hotels}")
    print(f"🔍 寻找更便宜的酒店...")
    
    # 调试：显示所有候选酒店
    print(f"🔍 所有候选酒店:")
    for i, hotel in enumerate(hotels_data):
        hotel_name = hotel.get("酒店名称", "")
        hotel_price = _extract_hotel_price(hotel.get("价格", "999元"))
        excluded_status = "🚫已排除" if hotel_name in excluded_hotels else "✅可用"
        print(f"  {i+1}. {hotel_name} - {hotel_price}元/晚 {excluded_status}")
    
    # 筛选出比当前酒店更便宜且不在排除列表中的酒店
    cheaper_hotels = []
    for hotel in hotels_data:
        hotel_name = hotel.get("酒店名称", "")
        hotel_price = _extract_hotel_price(hotel.get("价格", "999元"))
        
        # 必须不在排除列表中且价格更便宜
        if hotel_name not in excluded_hotels and hotel_price < current_price:
            cheaper_hotels.append((hotel, hotel_price))
            print(f"  候选: {hotel_name} - {hotel_price}元/晚 (节省{current_price - hotel_price}元)")
        elif hotel_name in excluded_hotels:
            print(f"  跳过(已排除): {hotel_name} - {hotel_price}元/晚")
    
    if cheaper_hotels:
        # 在更便宜的酒店中选择评分最高的
        cheaper_hotels.sort(key=lambda x: (x[1], -float(x[0].get("评分", "0"))))  # 按价格升序，评分降序
        selected_hotel = cheaper_hotels[0][0]
        selected_price = cheaper_hotels[0][1]
        
        print(f"✅ 找到更便宜的酒店: {selected_hotel['酒店名称']}")
        print(f"   价格: {selected_price}元/晚 (节省{current_price - selected_price}元/晚)")
        print(f"   评分: {selected_hotel['评分']}")
        
        return selected_hotel
    else:
        # 如果没有更便宜的酒店，选择不在排除列表中的最便宜酒店
        available_hotels = [h for h in hotels_data if h.get("酒店名称", "") not in excluded_hotels]
        if available_hotels:
            cheapest_hotel = min(available_hotels, key=lambda x: _extract_hotel_price(x.get("价格", "999元")))
            cheapest_price = _extract_hotel_price(cheapest_hotel.get("价格", "999元"))
            
            print(f"✅ 选择未排除的最便宜酒店: {cheapest_hotel['酒店名称']} ({cheapest_price}元/晚)")
            if cheapest_price >= current_price:
                print(f"⚠️ 注意：该酒店价格({cheapest_price}元)不低于当前酒店({current_price}元)")
            return cheapest_hotel
        else:
            print(f"⚠️ 所有酒店都已被排除，保持当前选择")
            return current_hotel

def _extract_hotel_price(price_str: str) -> float:
    """从价格字符串中提取数字"""
    try:
        return float(''.join(filter(str.isdigit, price_str)))
    except:
        return 999.0  # 默认价格

def hotel_optimization(state: AgentState) -> AgentState:
    """
    酒店优化节点 - 专门处理酒店优化，不重新执行交通规划和强度计算
    
    功能：
    1. 从已有酒店搜索结果中选择更便宜的酒店
    2. 更新酒店选择
    3. 重新计算预算（基于新的酒店价格）
    """
    print("🏨 执行酒店优化...")
    
    hotel_optimization_attempts = state.get("hotel_optimization_attempts", 0)
    max_hotel_optimization_attempts = state.get("max_hotel_optimization_attempts", 1)
    
    print(f"🔍 当前优化次数: {hotel_optimization_attempts}")
    print(f"🔍 最大优化次数: {max_hotel_optimization_attempts}")
    
    # 检查是否超过最大优化次数
    if hotel_optimization_attempts >= max_hotel_optimization_attempts:
        print(f"❌ 已达到最大优化次数({max_hotel_optimization_attempts})，无法继续优化")
        return state
    
    # 获取已有酒店搜索结果
    existing_results = state.get("hotel_search_results", [])
    if not existing_results:
        print("❌ 未找到已有酒店搜索结果，无法进行优化")
        return state
    
    print(f"✅ 使用{len(existing_results)}个已有酒店候选进行优化选择")
    
    # 将当前选择的酒店添加到排除列表
    current_hotels = state.get("selected_hotels", [])
    if current_hotels:
        current_hotel_name = current_hotels[0].get("酒店名称", "")
        if current_hotel_name and current_hotel_name not in state["excluded_hotels"]:
            state["excluded_hotels"].append(current_hotel_name)
            print(f"🚫 将当前酒店加入排除列表: {current_hotel_name}")
    
    # 显示排除列表
    excluded_hotels = state.get("excluded_hotels", [])
    if excluded_hotels:
        print(f"🚫 已排除的酒店: {', '.join(excluded_hotels)}")
    
    # 确保排除列表被正确设置
    if "excluded_hotels" not in state:
        state["excluded_hotels"] = []
    
    # 选择更便宜的酒店（确保排除列表被正确传递）
    print(f"🔍 开始选择酒店，排除列表: {state['excluded_hotels']}")
    selected_hotel = _select_cheaper_hotel(state, existing_results)
    selection_reason = f"第{hotel_optimization_attempts + 1}次优化，选择更便宜酒店"
    
    # 增加优化尝试次数
    state["hotel_optimization_attempts"] = hotel_optimization_attempts + 1
    
    # 更新酒店选择
    state["selected_hotels"] = [selected_hotel]
    
    print(f"✅ 优化选择酒店: {selected_hotel['酒店名称']}")
    print(f"   评分: {selected_hotel['评分']}")
    print(f"   房型: {selected_hotel['房型']}")
    print(f"   价格: {selected_hotel['价格']}")
    print(f"   选择原因: {selection_reason}")
    
    # 添加酒店选择记录
    if "hotel_selection_history" not in state:
        state["hotel_selection_history"] = []
    
    state["hotel_selection_history"].append({
        "selected_hotel": selected_hotel,
        "selection_reason": selection_reason,
        "selection_time": f"optimization_{hotel_optimization_attempts + 1}",
        "available_options": len(existing_results),
        "optimization_attempt": hotel_optimization_attempts + 1
    })
    
    # 重要：清除之前的交通规划结果，确保使用新酒店重新规划
    if "transportation_plans" in state:
        old_hotel = state["transportation_plans"].get("hotel_used", "未知")
        del state["transportation_plans"]
        print(f"🔄 已清除之前的交通规划（酒店: {old_hotel}），将使用新酒店（{selected_hotel['酒店名称']}）重新规划")
    
    # 同时清除其他相关状态，确保完全重新计算
    for key in ["valid_transport_plans", "calculated_intensity", "intensity_satisfied"]:
        if key in state:
            del state[key]
            print(f"🔄 已清除{key}状态，确保重新计算")
    
    # 重新计算预算（基于新的酒店价格）
    print("💰 重新计算预算...")
    
    # 获取基础信息
    info = state.get("structured_info", {})
    group = info.get("group", {})
    total_people = group.get("adults", 1) + group.get("children", 0)
    room_requirements = state.get("room_requirements", 1)
    daily_candidates = state.get("daily_candidates", [])
    trip_days = len(daily_candidates)
    
    # 计算新的酒店费用
    hotel_price_per_night = _extract_hotel_price(selected_hotel.get("价格", "0元"))
    total_hotel_cost = hotel_price_per_night * room_requirements * trip_days
    
    # 获取之前的费用信息
    cost_breakdown = state.get("cost_breakdown", {})
    ticket_cost = cost_breakdown.get("ticket_cost", 0)
    transport_cost = cost_breakdown.get("transport_cost", 0)
    
    # 如果没有之前的费用信息，从推荐方案中获取
    if ticket_cost == 0 or transport_cost == 0:
        recommended_plan = state.get("recommended_plan", {})
        if recommended_plan:
            ticket_cost = recommended_plan.get("ticket_cost", 0)
            transport_cost = recommended_plan.get("transport_cost", 0)
            print(f"🔍 从推荐方案获取费用信息: 门票{ticket_cost}元, 交通{transport_cost}元")
    
    # 计算新的总费用
    new_total_cost = ticket_cost + total_hotel_cost + transport_cost
    
    # 更新费用信息
    state["calculated_cost"] = new_total_cost
    state["cost_breakdown"]["hotel_cost"] = total_hotel_cost
    state["cost_breakdown"]["hotel_details"] = {
        "hotel_name": selected_hotel.get("酒店名称", ""),
        "price_per_night": hotel_price_per_night,
        "rooms": room_requirements,
        "nights": trip_days
    }
    
    # 检查预算是否满足
    budget_info = info.get("budget", {})
    budget_limit = budget_info.get("total") or (budget_info.get("per_day", 1000) * trip_days)
    state["budget_satisfied"] = new_total_cost <= budget_limit
    
    print(f"💰 优化后费用:")
    print(f"   景点门票: {ticket_cost}元")
    print(f"   酒店住宿: {total_hotel_cost}元")
    print(f"   交通费用: {transport_cost}元")
    print(f"   总费用: {new_total_cost}元")
    print(f"   预算限制: {budget_limit}元")
    print(f"   预算状态: {'✅ 满足' if state['budget_satisfied'] else '❌ 超出'}")
    
    print("✅ 酒店优化完成")
    return state

# 6. 预算检查节点 - budget_check
def budget_check(state: AgentState) -> AgentState:
    """
    预算检查节点 - 检查推荐方案是否符合预算约束
    
    功能：
    1. 检查当前推荐方案是否符合预算
    2. 如果不符合且未尝试过酒店优化，返回优化标记
    3. 输出预算检查结果
    """
    print("💳 执行预算检查...")
    
    # 获取预算计算结果
    recommended_plan = state.get("recommended_plan", {})
    budget_satisfied = state.get("budget_satisfied", False)
    hotel_optimization_attempts = state.get("hotel_optimization_attempts", 0)
    
    info = state.get("structured_info", {})
    budget_info = info.get("budget", {})
    daily_candidates = state.get("daily_candidates", [])
    trip_days = len(daily_candidates)
    budget_limit = budget_info.get("total") or (budget_info.get("per_day", 1000) * trip_days)
    
    if not recommended_plan:
        print("❌ 未找到推荐方案，无法进行预算检查")
        state["budget_satisfied"] = False
        return state
    
    total_cost = recommended_plan.get("total_cost", 0)
    strategy = recommended_plan.get("strategy", "未知")
    
    print(f"\n💰 预算检查详情:")
    print(f"  推荐方案: {strategy}")
    print(f"  总费用: {total_cost}元")
    print(f"  预算限制: {budget_limit}元")
    print(f"  酒店优化尝试次数: {hotel_optimization_attempts}")
    
    if budget_satisfied:
        print(f"✅ 预算检查通过！剩余预算: {budget_limit - total_cost}元")
        state["budget_check_result"] = "满足预算"
    else:
        exceed_amount = total_cost - budget_limit
        print(f"❌ 预算检查未通过！超出预算: {exceed_amount}元")
        
        if hotel_optimization_attempts == 0:
            print("💡 将尝试选择更便宜的酒店来降低成本")
            state["budget_check_result"] = "需要优化酒店"
        else:
            print("⚠️ 已尝试酒店优化一次，但仍超出预算")
            state["budget_check_result"] = "优化后仍超预算"
    
    print("✅ 预算检查完成")
    return state

# 7. 最终输出节点 - final_output
def final_output(state: AgentState) -> AgentState:
    """
    最终输出节点 - 输出完整的旅行方案
    
    功能：
    1. 输出最终的景点安排
    2. 输出选择的酒店信息
    3. 输出推荐的交通方式
    4. 输出费用汇总
    5. 输出预算状态
    """
    print("\n" + "="*80)
    print("🎉 北京旅行方案最终输出")
    print("="*80)
    
    # 获取基础信息
    info = state.get("structured_info", {})
    recommended_plan = state.get("recommended_plan", {})
    daily_candidates = state.get("daily_candidates", [])
    selected_hotels = state.get("selected_hotels", [])
    budget_satisfied = state.get("budget_satisfied", False)
    budget_check_result = state.get("budget_check_result", "未检查")
    hotel_optimization_attempts = state.get("hotel_optimization_attempts", 0)
    
    # 基础行程信息
    start_date = info.get("start_date", "未知")
    end_date = info.get("end_date", "未知") 
    group = info.get("group", {})
    total_people = group.get("adults", 1) + group.get("children", 0) + group.get("elderly", 0)
    
    print(f"\n📅 行程基本信息:")
    print(f"  出行日期: {start_date} 至 {end_date}")
    print(f"  出行人数: {total_people}人 (成人{group.get('adults', 1)}人, 儿童{group.get('children', 0)}人, 老人{group.get('elderly', 0)}人)")
    print(f"  行程天数: {len(daily_candidates)}天")
    
    # 1. 景点安排
    print(f"\n🎯 每日景点安排:")
    if daily_candidates:
        for i, day_info in enumerate(daily_candidates, 1):
            date = day_info.get("date", f"第{i}天")
            pois = day_info.get("pois", [])
            
            print(f"\n  📍 第{i}天 ({date}):")
            if pois:
                for j, poi in enumerate(pois, 1):
                    poi_name = poi.get("name", "未知景点")
                    duration = poi.get("suggested_duration_hours", 2.0)
                    ticket_price = _get_poi_ticket_price(poi)
                    print(f"    {j}. {poi_name} (游玩时长: {duration}小时, 门票: {ticket_price}元/人)")
            else:
                print(f"    暂无景点安排")
    else:
        print("  ❌ 未找到景点安排")
    
    # 2. 酒店信息
    print(f"\n🏨 酒店安排:")
    if selected_hotels:
        hotel = selected_hotels[0]
        hotel_name = hotel.get("酒店名称", "未知酒店")
        hotel_rating = hotel.get("评分", "未知")
        hotel_room_type = hotel.get("房型", "未知")
        hotel_price = hotel.get("价格", "未知")
        
        print(f"  酒店名称: {hotel_name}")
        print(f"  评分: {hotel_rating}")
        print(f"  房型: {hotel_room_type}")
        print(f"  价格: {hotel_price}")
        
        if hotel_optimization_attempts > 0:
            print(f"  💡 经过{hotel_optimization_attempts}次优化选择")
    else:
        print("  ❌ 未找到酒店安排")
    
    # 3. 交通方式
    print(f"\n🚗 推荐交通方案:")
    if recommended_plan:
        strategy = recommended_plan.get("strategy", "未知")
        transport_cost = recommended_plan.get("transport_cost", 0)
        
        print(f"  推荐方案: {strategy}")
        print(f"  交通费用: {transport_cost}元")
        
        # 显示交通详情
        transportation_plans = state.get("transportation_plans", {})
        plan_name = recommended_plan.get("plan_name", "")
        if plan_name in transportation_plans:
            transport_detail = transportation_plans[plan_name]
            daily_plans = transport_detail.get("daily_plans", [])
            
            for day_plan in daily_plans:
                date = day_plan.get("date", "")
                routes = day_plan.get("routes", [])
                print(f"\n    📅 {date}:")
                for route in routes:
                    segment = route.get("segment", "")
                    method = route.get("method", "")
                    cost = route.get("cost", "")
                    print(f"      {segment}: {method} ({cost})")
    else:
        print("  ❌ 未找到交通方案")
    
    # 4. 费用汇总
    print(f"\n💰 费用汇总:")
    if recommended_plan:
        ticket_cost = recommended_plan.get("ticket_cost", 0)
        hotel_cost = recommended_plan.get("hotel_cost", 0)
        transport_cost = recommended_plan.get("transport_cost", 0)
        total_cost = recommended_plan.get("total_cost", 0)
        
        print(f"  🎫 景点门票: {ticket_cost}元")
        print(f"  🏨 酒店住宿: {hotel_cost}元")
        print(f"  🚗 交通费用: {transport_cost}元")
        print(f"  💯 总费用: {total_cost}元")
        
        # 预算状态
        budget_info = info.get("budget", {})
        budget_limit = budget_info.get("total") or (budget_info.get("per_day", 1000) * len(daily_candidates))
        
        print(f"\n📊 预算状态:")
        print(f"  预算限制: {budget_limit}元")
        print(f"  检查结果: {budget_check_result}")
        
        if budget_satisfied:
            remaining = budget_limit - total_cost
            print(f"  ✅ 符合预算，剩余: {remaining}元")
        else:
            exceed = total_cost - budget_limit
            print(f"  ❌ 超出预算: {exceed}元")
            if hotel_optimization_attempts > 0:
                print(f"  💡 已尝试{hotel_optimization_attempts}次酒店优化")
    else:
        print("  ❌ 未找到费用信息")
    
    # 5. 优化建议
    if not budget_satisfied:
        print(f"\n💡 预算优化建议:")
        print(f"  1. 选择更便宜的酒店或降低住宿标准")
        print(f"  2. 减少景点数量或选择免费景点")
        print(f"  3. 多使用公共交通，减少出租车")
        print(f"  4. 调整行程天数")
    
    print("\n" + "="*80)
    print("🎊 感谢使用北京旅行规划助手！祝您旅途愉快！")
    print("="*80)
    
    return state

def _get_default_ticket_price(poi_name: str) -> float:
    """根据景点名称获取默认门票价格"""
    # 移除常见的后缀词
    clean_name = poi_name
    for suffix in ["博物馆", "博物院", "景区", "公园", "风景区", "旅游区", "度假村"]:
        clean_name = clean_name.replace(suffix, "")
    
    # 知名景点的具体价格
    known_prices = {
        "故宫": 60, "故宫博物院": 60,
        "天安门": 15, 
        "颐和园": 30,
        "长城": 45, "八达岭长城": 45, "慕田峪长城": 45,
        "天坛": 15,
        "圆明园": 25,
        "北海公园": 10,
        "景山公园": 2,
        "雍和宫": 25,
        "孔庙": 30,
        "恭王府": 40,
        "明十三陵": 45,
        "鸟巢": 50, "国家体育场": 50,
        "水立方": 30, "国家游泳中心": 30
    }
    
    # 精确匹配
    if clean_name in known_prices:
        return known_prices[clean_name]
    
    # 模糊匹配
    for name, price in known_prices.items():
        if name in poi_name or poi_name in name:
            return price
    
    # 根据景点类型给出默认价格
    if any(word in poi_name for word in ["博物馆", "博物院"]):
        return 20  # 博物馆类
    elif any(word in poi_name for word in ["公园", "园"]):
        return 10  # 公园类
    elif any(word in poi_name for word in ["寺", "庙", "宫"]):
        return 25  # 宗教建筑
    elif any(word in poi_name for word in ["长城", "城墙"]):
        return 45  # 长城类
    else:
        return 30  # 通用默认价格

# ==================== 辅助函数 ====================

def determine_daily_time_budget(group):
    from .poi_utils import determine_daily_time_budget as _determine_daily_time_budget
    return _determine_daily_time_budget(group)

def compute_trip_days(start_date, end_date):
    from .poi_utils import compute_trip_days as _compute_trip_days
    return _compute_trip_days(start_date, end_date)

def _calculate_plan_intensity_simple(daily_candidates, plan_data):
    """计算单个交通方案的强度，以小时为单位"""
    strategy = plan_data.get("strategy", "未知方案")
    daily_plans = plan_data.get("daily_plans", [])
    
    daily_details = []
    total_hours = 0
    
    for transport_day in daily_plans:
        # 获取交通日期
        transport_date = transport_day.get("date", "")
        day_idx = transport_day.get("day", 0)
        
        # 在daily_candidates中查找对应日期的POI数据
        poi_day = None
        for candidate_day in daily_candidates:
            if candidate_day.get("date") == transport_date:
                poi_day = candidate_day
                break
        
        if poi_day is None:
            # 备用：按索引查找
            for j, candidate_day in enumerate(daily_candidates):
                if j + 1 == day_idx:
                    poi_day = candidate_day
                    break
        
        # 如果还是没找到，跳过这一天
        if poi_day is None:
            print(f"⚠️ 未找到日期 {transport_date} 的POI数据")
            continue
            
        # 计算景点时间
        poi_list = poi_day.get("pois", [])
        poi_hours = sum(poi.get("suggested_duration_hours", 2.0) for poi in poi_list)
        
        # 获取交通时间（分钟转小时）
        transport_minutes = transport_day.get("day_total_time", 0)
        transport_hours = transport_minutes / 60.0
        
        # 当日总时间
        daily_total_hours = poi_hours + transport_hours
        
        daily_details.append({
            "date": transport_date,
            "day": day_idx,
            "poi_hours": poi_hours,
            "transport_hours": transport_hours,
            "total_hours": daily_total_hours,
            "poi_count": len(poi_list)
        })
        
        total_hours += daily_total_hours
    
    # 计算平均每日时间
    avg_daily_hours = total_hours / len(daily_details) if daily_details else 0
    
    return {
        "strategy": strategy,
        "daily_details": daily_details,
        "total_hours": total_hours,
        "avg_daily_hours": avg_daily_hours
    }

def _print_intensity_simple(plan_name, result):
    """输出简化的强度计算结果"""
    strategy = result.get("strategy", plan_name)
    total_hours = result.get("total_hours", 0)
    avg_daily_hours = result.get("avg_daily_hours", 0)
    daily_details = result.get("daily_details", [])
    
    print(f"\n📊 【{strategy}】强度计算:")
    print(f"   总时长: {total_hours:.1f}小时")
    print(f"   日均时长: {avg_daily_hours:.1f}小时")
    
    for day in daily_details:
        print(f"   {day['date']}: POI游玩{day['poi_hours']:.1f}h + 交通{day['transport_hours']:.1f}h = {day['total_hours']:.1f}h")
    
    print(f"   详细信息: {len(daily_details)}天行程安排")