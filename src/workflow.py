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
        
        # 预算相关初始化
        "calculated_cost": 0.0,
        "cost_breakdown": {},
        "budget_satisfied": True,
        "budget_optimization_target": "",
        
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
        "daily_available_pois": []
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
    workflow.add_node("intensity_check", intensity_check)
    workflow.add_node("opt_intensity", opt_intensity)
    workflow.add_node("restaurant_selection", restaurant_selection)
    workflow.add_node("budget_calculate", budget_calculate)
    workflow.add_node("budget_check1", budget_check1)
    workflow.add_node("select_budget_adjustment_target", select_budget_adjustment_target)
    workflow.add_node("opt_hotel", opt_hotel)
    workflow.add_node("hotel_selection_apply", hotel_selection_apply)
    workflow.add_node("intensity_calculate2", intensity_calculate2)
    workflow.add_node("intensity_check2", intensity_check2)
    workflow.add_node("budget_check4", budget_check4)
    workflow.add_node("opt_transportation", opt_transportation)
    workflow.add_node("budget_check3", budget_check3)
    workflow.add_node("opt_restaurant", opt_restaurant)
    workflow.add_node("budget_check2", budget_check2)
    
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
    workflow.add_edge("intensity_calculate", "intensity_check")
    
    # intensity_check的条件边
    def decide_from_intensity_check(state: AgentState) -> str:
        intensity_satisfied = state.get("intensity_satisfied", True)
        if intensity_satisfied:
            return "restaurant_selection"
        else:
            return "opt_intensity"
    
    workflow.add_conditional_edges(
        "intensity_check",
        decide_from_intensity_check,
        {
            "restaurant_selection": "restaurant_selection",
            "opt_intensity": "opt_intensity"
        }
    )
    
    # opt_intensity的条件边
    def decide_from_opt_intensity(state: AgentState) -> str:
        can_optimize = state.get("can_optimize_intensity", False)
        if can_optimize:
            return "hotel_selection"  # 回到hotel_selection重新开始
        else:
            return END  # 结束，提醒用户尝试更换酒店位置或者景点
    
    workflow.add_conditional_edges(
        "opt_intensity",
        decide_from_opt_intensity,
        {
            "hotel_selection": "hotel_selection",
            END: END
        }
    )
    
    workflow.add_edge("restaurant_selection", "budget_calculate")
    workflow.add_edge("budget_calculate", "budget_check1")
    
    # budget_check1的条件边
    def decide_from_budget_check1(state: AgentState) -> str:
        budget_satisfied = state.get("budget_satisfied", True)
        if budget_satisfied:
            return END  # 结束，生成最终行程
        else:
            # 检查是否所有优化方向都已标记为不可行
            hotel_blocked = state.get("hotel_optimization_blocked", False)
            transport_blocked = state.get("transport_optimization_blocked", False)
            restaurant_blocked = state.get("restaurant_optimization_blocked", False)
            
            if hotel_blocked and transport_blocked and restaurant_blocked:
                return END  # 结束，提醒用户提高预算
            else:
                return "select_budget_adjustment_target"
    
    workflow.add_conditional_edges(
        "budget_check1",
        decide_from_budget_check1,
        {
            "select_budget_adjustment_target": "select_budget_adjustment_target",
            END: END
        }
    )
    
    # select_budget_adjustment_target的条件边
    def decide_budget_optimization_target(state: AgentState) -> str:
        optimization_target = state.get("budget_optimization_target", "")
        if optimization_target == "hotel":
            return "opt_hotel"
        elif optimization_target == "transportation":
            return "opt_transportation"
        elif optimization_target == "restaurant":
            return "opt_restaurant"
        else:
            return END  # 没有可优化目标，结束
    
    workflow.add_conditional_edges(
        "select_budget_adjustment_target",
        decide_budget_optimization_target,
        {
            "opt_hotel": "opt_hotel",
            "opt_transportation": "opt_transportation", 
            "opt_restaurant": "opt_restaurant",
            END: END
        }
    )
    
    # 酒店优化路径
    workflow.add_edge("opt_hotel", "hotel_selection_apply")
    workflow.add_edge("hotel_selection_apply", "transportation_planning")  # 重规划交通
    # transportation_planning连接到intensity_calculate2
    
    # 需要添加一个条件边来区分第一次和第二次intensity_calculate
    def decide_after_transportation(state: AgentState) -> str:
        is_optimization_round = state.get("is_optimization_round", False)
        if is_optimization_round:
            return "intensity_calculate2"
        else:
            return "intensity_calculate"
    
    # 更新transportation_planning的连接
    workflow.add_conditional_edges(
        "transportation_planning",
        decide_after_transportation,
        {
            "intensity_calculate": "intensity_calculate",
            "intensity_calculate2": "intensity_calculate2"
        }
    )
    
    workflow.add_edge("intensity_calculate2", "intensity_check2")
    
    # intensity_check2的条件边
    def decide_from_intensity_check2(state: AgentState) -> str:
        intensity_satisfied = state.get("intensity_satisfied", True)
        if intensity_satisfied:
            return "budget_check4"
        else:
            # 标记酒店方向暂不可行，返回动态决策
            state["hotel_optimization_blocked"] = True
            return "select_budget_adjustment_target"
    
    workflow.add_conditional_edges(
        "intensity_check2",
        decide_from_intensity_check2,
        {
            "budget_check4": "budget_check4",
            "select_budget_adjustment_target": "select_budget_adjustment_target"
        }
    )
    
    # budget_check4的条件边
    def decide_from_budget_check4(state: AgentState) -> str:
        budget_satisfied = state.get("budget_satisfied", True)
        if budget_satisfied:
            return END  # 成功，生成最终行程
        else:
            # 标记酒店方向暂不可行，返回动态决策
            state["hotel_optimization_blocked"] = True
            return "select_budget_adjustment_target"
    
    workflow.add_conditional_edges(
        "budget_check4",
        decide_from_budget_check4,
        {
            "select_budget_adjustment_target": "select_budget_adjustment_target",
            END: END
        }
    )
    
    # 交通优化路径
    workflow.add_edge("opt_transportation", "budget_check3")
    
    # budget_check3的条件边
    def decide_from_budget_check3(state: AgentState) -> str:
        budget_satisfied = state.get("budget_satisfied", True)
        if budget_satisfied:
            return END  # 成功，生成最终行程
        else:
            # 标记交通方向暂不可行，返回动态决策
            state["transport_optimization_blocked"] = True
            return "select_budget_adjustment_target"
    
    workflow.add_conditional_edges(
        "budget_check3",
        decide_from_budget_check3,
        {
            "select_budget_adjustment_target": "select_budget_adjustment_target",
            END: END
        }
    )
    
    # 餐厅优化路径
    workflow.add_edge("opt_restaurant", "budget_check2")
    
    # budget_check2的条件边
    def decide_from_budget_check2(state: AgentState) -> str:
        budget_satisfied = state.get("budget_satisfied", True)
        if budget_satisfied:
            return END  # 成功，生成最终行程
        else:
            # 标记餐厅方向暂不可行，返回动态决策
            state["restaurant_optimization_blocked"] = True
            return "select_budget_adjustment_target"
    
    workflow.add_conditional_edges(
        "budget_check2",
        decide_from_budget_check2,
        {
            "select_budget_adjustment_target": "select_budget_adjustment_target",
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
        daily_time_limit = 11  # 有老人或儿童，每天最多11小时
    else:
        daily_time_limit = 14  # 只有成年人，每天最多12小时
    
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
    print("🏨 执行酒店选择...")
    # TODO: 实现酒店选择逻辑
    
    info = state.get("structured_info", {})
    room_requirements = state.get("room_requirements", 1)
    start_date = info.get("start_date")
    end_date = info.get("end_date")
    
    state["selected_hotels"] = []  # 临时占位
    return state

# 3. 交通规划节点 - transportation_planning  
def transportation_planning(state: AgentState) -> AgentState:
    """交通规划(生成多方案)"""
    print("🚗 执行交通规划...")
    # TODO: 实现交通规划逻辑
    
    daily_candidates = state.get("daily_candidates", [])
    selected_hotels = state.get("selected_hotels", [])
    
    state["transportation_plan"] = []  # 临时占位
    return state

# 4. 强度计算节点 - intensity_calculate
def intensity_calculate(state: AgentState) -> AgentState:
    """强度计算"""
    print("💪 执行强度计算...")
    # TODO: 实现强度计算逻辑
    
    transportation_plan = state.get("transportation_plan", [])
    
    state["calculated_intensity"] = 0  # 临时占位
    return state

# 5. 强度检查节点 - intensity_check
def intensity_check(state: AgentState) -> AgentState:
    """是否满足强度"""
    print("✅ 执行强度检查...")
    # TODO: 实现强度检查逻辑
    
    calculated_intensity = state.get("calculated_intensity", 0)
    daily_time_limit = state.get("daily_time_limit", 12)
    
    # 简单的临时逻辑
    intensity_satisfied = calculated_intensity <= daily_time_limit
    state["intensity_satisfied"] = intensity_satisfied
    
    if intensity_satisfied:
        print("✅ 强度检查通过")
    else:
        print("❌ 强度检查未通过")
    
    return state

# 6. 强度优化节点 - opt_intensity
def opt_intensity(state: AgentState) -> AgentState:
    """强度优化可继续?(最多优化1次)"""
    print("🔧 执行强度优化...")
    # TODO: 实现强度优化逻辑
    
    optimization_attempts = state.get("intensity_optimization_attempts", 0)
    
    # 最多优化1次
    if optimization_attempts < 1:
        state["intensity_optimization_attempts"] = optimization_attempts + 1
        state["can_optimize_intensity"] = True
        print("✅ 可以进行强度优化")
    else:
        state["can_optimize_intensity"] = False
        print("❌ 已达到强度优化次数上限")
    
    return state

# 7. 餐厅选择节点 - restaurant_selection
def restaurant_selection(state: AgentState) -> AgentState:
    """餐厅选择"""
    print("🍽️ 执行餐厅选择...")
    # TODO: 实现餐厅选择逻辑
    
    daily_candidates = state.get("daily_candidates", [])
    info = state.get("structured_info", {})
    preferences = info.get("preferences", {})
    cuisine_prefs = preferences.get("cuisine", [])
    
    state["selected_restaurants"] = []  # 临时占位
    return state

# 8. 预算计算节点 - budget_calculate
def budget_calculate(state: AgentState) -> AgentState:
    """预算检查"""
    print("💰 执行预算计算...")
    # TODO: 实现预算计算逻辑
    
    selected_restaurants = state.get("selected_restaurants", [])
    selected_hotels = state.get("selected_hotels", [])
    transportation_plan = state.get("transportation_plan", [])
    
    state["calculated_cost"] = 0  # 临时占位
    state["cost_breakdown"] = {}  # 临时占位
    return state

# 9. 预算检查节点 - budget_check1  
def budget_check1(state: AgentState) -> AgentState:
    """是否满足预算"""
    print("💸 执行预算检查...")
    # TODO: 实现预算检查逻辑
    
    calculated_cost = state.get("calculated_cost", 0)
    info = state.get("structured_info", {})
    budget = info.get("budget", {})
    
    # 获取预算金额
    budget_amount = budget.get("total") or budget.get("per_day", 0) * state.get("trip_days", 1)
    
    # 简单的临时逻辑
    budget_satisfied = calculated_cost <= budget_amount
    state["budget_satisfied"] = budget_satisfied
    
    if budget_satisfied:
        print("✅ 预算检查通过")
    else:
        print("❌ 预算检查未通过")
    
    return state

# 10. 预算调整目标选择节点 - select_budget_adjustment_target
def select_budget_adjustment_target(state: AgentState) -> AgentState:
    """选择优化目标"""
    print("🎯 选择预算优化目标...")
    # TODO: 实现优化目标选择逻辑
    
    # 检查各个方向是否被阻塞
    hotel_blocked = state.get("hotel_optimization_blocked", False)
    transport_blocked = state.get("transport_optimization_blocked", False)  
    restaurant_blocked = state.get("restaurant_optimization_blocked", False)
    
    # 选择未被阻塞的优化方向，优先级：酒店 > 交通 > 餐厅
    if not hotel_blocked:
        state["budget_optimization_target"] = "hotel"
        print("🏨 选择酒店优化")
    elif not transport_blocked:
        state["budget_optimization_target"] = "transportation"
        print("🚗 选择交通优化")
    elif not restaurant_blocked:
        state["budget_optimization_target"] = "restaurant" 
        print("🍽️ 选择餐厅优化")
    else:
        state["budget_optimization_target"] = ""
        print("❌ 所有优化方向都已阻塞")
    
        return state
    
# 11. 酒店优化节点 - opt_hotel
def opt_hotel(state: AgentState) -> AgentState:
    """优化酒店"""
    print("🏨 执行酒店优化...")
    # TODO: 实现酒店优化逻辑
    
    state["optimized_hotels"] = []  # 临时占位
    return state

# 12. 酒店选择应用节点 - hotel_selection_apply
def hotel_selection_apply(state: AgentState) -> AgentState:
    """应用新酒店"""
    print("🏨 应用优化后的酒店...")
    # TODO: 实现酒店应用逻辑
    
    optimized_hotels = state.get("optimized_hotels", [])
    state["selected_hotels"] = optimized_hotels
    state["is_optimization_round"] = True  # 标记为优化轮次
    return state

# 13. 强度计算2节点 - intensity_calculate2
def intensity_calculate2(state: AgentState) -> AgentState:
    """强度检查2"""
    print("💪 执行强度计算2...")
    # TODO: 实现强度计算逻辑（与intensity_calculate相同）
    
    transportation_plan = state.get("transportation_plan", [])
    
    state["calculated_intensity"] = 0  # 临时占位
    return state

# 14. 强度检查2节点 - intensity_check2
def intensity_check2(state: AgentState) -> AgentState:
    """是否满足强度2"""
    print("✅ 执行强度检查2...")
    # TODO: 实现强度检查逻辑（与intensity_check相同）
    
    calculated_intensity = state.get("calculated_intensity", 0)
    daily_time_limit = state.get("daily_time_limit", 12)
    
    # 简单的临时逻辑
    intensity_satisfied = calculated_intensity <= daily_time_limit
    state["intensity_satisfied"] = intensity_satisfied
    
    if intensity_satisfied:
        print("✅ 强度检查2通过")
    else:
        print("❌ 强度检查2未通过")
    
    return state

# 15. 预算检查4节点 - budget_check4
def budget_check4(state: AgentState) -> AgentState:
    """预算是否合格4"""
    print("💸 执行预算检查4...")
    # TODO: 实现预算检查逻辑（与budget_check1相同）
    
    calculated_cost = state.get("calculated_cost", 0)
    info = state.get("structured_info", {})
    budget = info.get("budget", {})
    
    # 获取预算金额
    budget_amount = budget.get("total") or budget.get("per_day", 0) * state.get("trip_days", 1)
    
    # 简单的临时逻辑
    budget_satisfied = calculated_cost <= budget_amount
    state["budget_satisfied"] = budget_satisfied
    
    if budget_satisfied:
        print("✅ 预算检查4通过")
    else:
        print("❌ 预算检查4未通过")
    
    return state

# 16. 交通优化节点 - opt_transportation
def opt_transportation(state: AgentState) -> AgentState:
    """优化交通方式"""
    print("🚗 执行交通优化...")
    # TODO: 实现交通优化逻辑
    
    current_plan = state.get("transportation_plan", [])
    state["optimized_transportation_plan"] = current_plan  # 临时占位
    return state

# 17. 预算检查3节点 - budget_check3
def budget_check3(state: AgentState) -> AgentState:
    """预算是否合格3"""
    print("💸 执行预算检查3...")
    # TODO: 实现预算检查逻辑（与budget_check1相同）
    
    calculated_cost = state.get("calculated_cost", 0)
    info = state.get("structured_info", {})
    budget = info.get("budget", {})
    
    # 获取预算金额
    budget_amount = budget.get("total") or budget.get("per_day", 0) * state.get("trip_days", 1)
    
    # 简单的临时逻辑
    budget_satisfied = calculated_cost <= budget_amount
    state["budget_satisfied"] = budget_satisfied
    
    if budget_satisfied:
        print("✅ 预算检查3通过")
    else:
        print("❌ 预算检查3未通过")
    
    return state

# 18. 餐厅优化节点 - opt_restaurant
def opt_restaurant(state: AgentState) -> AgentState:
    """优化餐厅"""
    print("🍽️ 执行餐厅优化...")
    # TODO: 实现餐厅优化逻辑
    
    current_restaurants = state.get("selected_restaurants", [])
    state["optimized_restaurants"] = current_restaurants  # 临时占位
    return state

# 19. 预算检查2节点 - budget_check2
def budget_check2(state: AgentState) -> AgentState:
    """预算是否合格2"""
    print("💸 执行预算检查2...")
    # TODO: 实现预算检查逻辑（与budget_check1相同）
    
    calculated_cost = state.get("calculated_cost", 0)
    info = state.get("structured_info", {})
    budget = info.get("budget", {})
    
    # 获取预算金额
    budget_amount = budget.get("total") or budget.get("per_day", 0) * state.get("trip_days", 1)
    
    # 简单的临时逻辑
    budget_satisfied = calculated_cost <= budget_amount
    state["budget_satisfied"] = budget_satisfied
    
    if budget_satisfied:
        print("✅ 预算检查2通过")
    else:
        print("❌ 预算检查2未通过")
    
    return state

# 从poi_utils导入的函数
def determine_daily_time_budget(group):
    from .poi_utils import determine_daily_time_budget as _determine_daily_time_budget
    return _determine_daily_time_budget(group)

def compute_trip_days(start_date, end_date):
    from .poi_utils import compute_trip_days as _compute_trip_days
    return _compute_trip_days(start_date, end_date)
