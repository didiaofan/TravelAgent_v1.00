from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field, conint, validator
from datetime import datetime

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

# 定义状态结构
class AgentState(TypedDict):
    structured_info: Dict[str, Any]  # 已收集的结构化信息
    conversation: List[Dict[str, str]]  # 完整的对话历史
    missing_fields: List[str]  # 当前缺失的字段列表
    step_count: int  # 对话轮次计数器
