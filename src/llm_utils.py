from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from config import config

def create_woka_llm(temperature=0.5):
    """初始化沃卡平台兼容的LLM"""
    return ChatOpenAI(
        openai_api_base=config.WOKA_API_BASE,
        model_name=config.WOKA_MODEL_NAME,
        temperature=temperature,
        openai_api_key=config.OPENAI_API_KEY
    )

def create_parse_prompt():
    """创建解析用户输入的提示词模板"""
    return ChatPromptTemplate.from_template(
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

def create_parser(model_class):
    """创建Pydantic输出解析器"""
    return PydanticOutputParser(pydantic_object=model_class)
