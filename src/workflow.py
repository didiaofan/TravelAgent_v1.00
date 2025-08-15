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
        "step_count": 0
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

# 实时数据获取节点（占位）：天气/路程/票价等
def fetch_realtime_data(state: AgentState) -> AgentState:
    info = state.get("structured_info", {})
    info.setdefault("realtime", {})
    return state

# 约束校验节点（占位）：确保派生约束有效
def validate_constraints(state: AgentState) -> AgentState:
    info = state.get("structured_info", {})
    constraints = info.get("constraints", {}).get("derived", {})
    issues: List[str] = []
    daily = int(constraints.get("daily_time_budget_hours") or 0)
    days = int(constraints.get("trip_days") or 0)
    if daily <= 0:
        issues.append("invalid_daily_time_budget")
        constraints["daily_time_budget_hours"] = 12
    if days <= 0:
        issues.append("invalid_trip_days")
        constraints["trip_days"] = 1
    info.setdefault("validation", {})
    info["validation"]["issues"] = issues
    return state

# 生成候选景点节点（当信息完整且准备/校验后触发）
def generate_candidates(state: AgentState) -> AgentState:
    try:
        result = generate_candidate_attractions(state.get("structured_info", {}))
        state["structured_info"]["candidates"] = result.get("candidates", [])
        state["structured_info"]["daily_plan"] = result.get("daily_plan", [])
        state["structured_info"]["total_cost"] = result.get("total_cost")
        state["structured_info"]["itinerary_text"] = result.get("itinerary_text")
    except Exception:
        state["structured_info"].setdefault("candidates", [])
        state["structured_info"].setdefault("daily_plan", [])
        state["structured_info"].setdefault("total_cost", None)
        state["structured_info"].setdefault("itinerary_text", "")
    return state

# 生成追问节点
def generate_question(state: AgentState) -> AgentState:
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
    
    # 添加节点
    workflow.add_node("parse_input", parse_user_input)
    workflow.add_node("check_fields", check_missing_fields)
    workflow.add_node("ask_question", generate_question)
    workflow.add_node("prepare_constraints", prepare_constraints)
    workflow.add_node("fetch_realtime_data", fetch_realtime_data)
    workflow.add_node("validate_constraints", validate_constraints)
    workflow.add_node("generate_candidates", generate_candidates)
    
    # 设置入口点
    workflow.set_entry_point("parse_input")
    
    # 添加边
    workflow.add_edge("parse_input", "check_fields")
    
    # 条件边 - 决定下一步或结束
    def decide_next(state: AgentState) -> str:
        # 信息完整：进入准备阶段；否则继续追问；达到最大轮次直接结束
        if state["step_count"] >= MAX_CONVERSATION_STEPS:
            return END
        if not state["missing_fields"]:
            return "prepare_constraints"
        return "ask_question"
    
    workflow.add_conditional_edges(
        "check_fields",
        decide_next,
        {
            "ask_question": "ask_question",
            "prepare_constraints": "prepare_constraints",
            END: END
        }
    )
    
    # 准备 → 实时数据 → 校验 → 候选；完成后结束
    workflow.add_edge("prepare_constraints", "fetch_realtime_data")
    workflow.add_edge("fetch_realtime_data", "validate_constraints")
    workflow.add_edge("validate_constraints", "generate_candidates")
    workflow.add_edge("generate_candidates", END)
    # 从追问节点回到解析节点，但需要用户输入（由外层下一轮驱动）
    workflow.add_edge("ask_question", END)
    
    return workflow.compile()

# 从poi_utils导入的函数
def determine_daily_time_budget(group):
    from .poi_utils import determine_daily_time_budget as _determine_daily_time_budget
    return _determine_daily_time_budget(group)

def compute_trip_days(start_date, end_date):
    from .poi_utils import compute_trip_days as _compute_trip_days
    return _compute_trip_days(start_date, end_date)
