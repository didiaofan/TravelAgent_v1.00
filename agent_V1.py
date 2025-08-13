from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
import json
import os
import re
from pydantic import BaseModel, Field, conint, validator

# 结构化输出模型（用于约束LLM输出）
class GroupModel(BaseModel):
    adults: conint(ge=0) = 0
    children: conint(ge=0) = 0
    elderly: conint(ge=0) = 0

class BudgetModel(BaseModel):
    total: Optional[int] = None
    per_day: Optional[int] = None

class PreferencesModel(BaseModel):
    attraction_types: Optional[List[str]] = None
    must_visit: Optional[List[str]] = None
    cuisine: Optional[List[str]] = None
    avoid: Optional[List[str]] = None

class AgentExtraction(BaseModel):
    departure_city: Optional[str] = None
    destination_city: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget: Optional[BudgetModel] = None
    group: Optional[GroupModel] = None
    preferences: Optional[PreferencesModel] = None
    constraints: Optional[Dict[str, Any]] = None

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "sk-qxeM6fcmvcYArnoXdV7vMyJ1OUMHRlALVfbIn6y1AQn7VIxk"

# 定义状态结构
class AgentState(TypedDict):
    structured_info: Dict[str, Any]  # 已收集的结构化信息
    conversation: List[Dict[str, str]]  # 完整的对话历史
    missing_fields: List[str]  # 当前缺失的字段列表
    step_count: int  # 对话轮次计数器

# 必需的顶级字段及其子字段验证
REQUIRED_FIELDS = {
    "departure_city": lambda x: isinstance(x, str) and x.strip() != "",
    "destination_city": lambda x: isinstance(x, str) and x.strip() != "",
    "start_date": lambda x: isinstance(x, str) and len(x) == 10 and x.strip() != "" and x != "2023-10-01" and x != "2023-10-03",  # 避免默认日期和空字符串
    "end_date": lambda x: isinstance(x, str) and len(x) == 10 and x.strip() != "" and x != "2023-10-01" and x != "2023-10-03",  # 避免默认日期和空字符串
    "budget": lambda x: isinstance(x, dict) and (("total" in x and x["total"] > 0) or ("per_day" in x and x["per_day"] > 0)),  # 确保预算值大于0
    "group": lambda x: isinstance(x, dict) and all(k in x for k in ["adults", "children", "elderly"]) and any(v > 0 for k, v in x.items() if k in ["adults", "children", "elderly"]),
    "preferences": lambda x: isinstance(x, dict) and (
        (isinstance(x.get("attraction_types", []), list) and len([i for i in x.get("attraction_types", []) if str(i).strip() != ""]) > 0)
        or (isinstance(x.get("must_visit", []), list) and len([i for i in x.get("must_visit", []) if str(i).strip() != ""]) > 0)
        or (isinstance(x.get("cuisine", []), list) and len([i for i in x.get("cuisine", []) if str(i).strip() != ""]) > 0)
    )
}

# 最大对话轮次限制
MAX_CONVERSATION_STEPS = 10

# 初始化沃卡平台兼容的LLM
def create_woka_llm(temperature=0.5):
    return ChatOpenAI(
        openai_api_base="https://4.0.wokaai.com/v1",
        model_name="gpt-3.5-turbo",
        temperature=temperature
    )

# --- 帮助函数：从自由文本解析同行人数 ---
def extract_group_counts_from_text(free_text: str) -> Optional[Dict[str, int]]:
    if not isinstance(free_text, str) or not free_text.strip():
        return None
    text = free_text.strip()

    def find_count(keywords: list[str]) -> Optional[int]:
        for kw in keywords:
            # 关键字后跟数字 / 数字后跟关键字（中间允许最多3个非数字字符）
            pattern1 = re.compile(rf"{kw}[^\d]{0,3}(\d+)")
            pattern2 = re.compile(rf"(\d+)[^\d]{0,3}{kw}")
            m = pattern1.search(text)
            if m:
                return int(m.group(1))
            m = pattern2.search(text)
            if m:
                return int(m.group(1))
        return None

    adults = find_count(["成人", "大人", "成年人"])
    children = find_count(["儿童", "小孩", "孩子", "小朋友"])
    elderly = find_count(["老人", "长者", "老年", "老年人"])

    if adults is None and children is None and elderly is None:
        return None

    return {
        "adults": int(adults) if adults is not None else 0,
        "children": int(children) if children is not None else 0,
        "elderly": int(elderly) if elderly is not None else 0,
    }


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
        "step_count": 0  # 初始轮次为0
    }

# 解析用户输入节点
def parse_user_input(state: AgentState) -> AgentState:
    # 更新轮次计数器
    state["step_count"] += 1
    
    # 创建解析模板（使用Pydantic强约束）
    parser = PydanticOutputParser(pydantic_object=AgentExtraction)
    
    prompt = ChatPromptTemplate.from_template(
        "您是一个专业的旅行助手，需要从对话历史中提取结构化信息。\n"
        "当前已收集的信息:\n{current_info}\n\n"
        "最新用户输入:\n{new_input}\n\n"
        "请严格按以下JSON格式输出，仅包含用户提到的字段，不要添加额外字段：\n"
        "{format_instructions}\n"
        "重要注意事项:\n"
        "- destination_city默认是'北京'，除非用户特别指定\n"
        "- departure_city必须提取用户的出发城市（如'上海'、'太原'等）\n"
        "- start_date和end_date必须是YYYY-MM-DD格式的字符串，如果用户没有明确提供具体日期，请不要猜测或生成\n"
        "- group字段只有在用户明确提供了人数时才输出（如'成人2位，孩子1个，老人0'），不要从'孩子们'、'和家人'等模糊表述推断\n"
        "- budget字段必须包含total（总预算）或per_day（每日预算），根据用户提到的预算金额设置\n"
        "- 偏好中避免项默认是['无']\n"
        "- 请将departure_city、start_date、end_date放在根级别，不要嵌套在其他对象中\n"
        "- 严格遵循：只提取用户明确提到的信息，不要推测或生成任何未提供的信息"
    )
    
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
    
    # 设置入口点
    workflow.set_entry_point("parse_input")
    
    # 添加边
    workflow.add_edge("parse_input", "check_fields")
    
    # 条件边 - 决定下一步或结束
    def decide_next(state: AgentState) -> str:
        # 如果达到最大轮次或没有缺失字段，则结束
        if state["step_count"] >= MAX_CONVERSATION_STEPS or not state["missing_fields"]:
            return END
        return "ask_question"
    
    workflow.add_conditional_edges(
        "check_fields",
        decide_next,
        {
            "ask_question": "ask_question",
            END: END
        }
    )
    
    # 从追问节点回到解析节点，但需要用户输入
    # 这里我们不再自动循环，而是等待用户输入
    workflow.add_edge("ask_question", END)
    
    return workflow.compile()

# 主函数：运行Agent（单次执行）
def run_travel_agent(user_input: str):
    # 初始化状态
    state = init_state(user_input)
    workflow = create_agent_workflow()
    
    # 运行工作流
    try:
        # 执行工作流
        result = workflow.invoke(state)
        
        # 检查是否结束
        if END in result:
            return result[END]["structured_info"]
        
        # 如果没有结束，说明需要用户输入
        return result["structured_info"]
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("将使用当前信息生成行程...")
        return state["structured_info"]

# 多轮对话函数（避免递归）
def run_travel_agent_multi_turn(initial_input: str, max_turns: int = 5):
    """
    多轮对话版本，避免递归问题
    """
    # 初始化状态
    state = init_state(initial_input)
    workflow = create_agent_workflow()
    
    turn_count = 0
    
    while turn_count < max_turns:
        try:
            print(f"\n=== 第 {turn_count + 1} 轮对话 ===")
            
            # 执行工作流
            result = workflow.invoke(state)
            
            # 如果已无缺失字段或达到最大轮次，直接结束
            if (not result.get('missing_fields')) or (result.get('step_count', 0) >= MAX_CONVERSATION_STEPS):
                print("信息收集完成！")
                return result.get('structured_info', state['structured_info'])
            
            # 显示当前状态与缺失字段
            print(f"当前已收集信息: {json.dumps(result['structured_info'], ensure_ascii=False, indent=2)}")
            print(f"缺失字段: {result['missing_fields']}")
            
            # 仅当存在缺失字段时显示助手追问
            if result['missing_fields']:
                last_assistant_msg = None
                for msg in reversed(result.get('conversation', [])):
                    if msg.get('role') == 'assistant':
                        last_assistant_msg = msg
                        break
                if last_assistant_msg:
                    print(f"Assistant: {last_assistant_msg['content']}")
            
            # 获取用户输入（仅在缺失字段时）
            user_response = input("User: ")
            if user_response.lower() in ['quit', 'exit', '结束', '退出']:
                print("用户选择退出，使用当前信息生成行程。")
                return result['structured_info']
            
            # 更新状态，准备下一轮
            result['conversation'].append({"role": "user", "content": user_response})
            state = result
            turn_count += 1
            
        except Exception as e:
            print(f"发生错误: {e}")
            print("将使用当前信息生成行程...")
            return state["structured_info"]
    
    print("达到最大对话轮次，使用当前信息生成行程。")
    return state["structured_info"]

# 测试示例
if __name__ == "__main__":
    # user_input = "我带孩子从上海到北京玩两天，时间是2025-08-10至2025-08-12，想去故宫和环球影城，预算人均6000，喜欢历史和湘菜，有一个65岁的父亲同行。"
    user_input = "我带孩子们从上海到北京玩两天。"

    print("=== 旅行规划Agent V1 ===")
    print(f"User: {user_input}")
    
    # 使用多轮对话版本，避免递归问题
    final_info = run_travel_agent_multi_turn(user_input, max_turns=5)
    
    print("\n=== 结构化输出 ===")
    print(json.dumps(final_info, ensure_ascii=False, indent=2))