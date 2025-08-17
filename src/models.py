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
    
    # 约束处理阶段的数据
    candidate_pois: List[Dict[str, Any]]  # 候选景点列表
    weather_adjusted_pois: List[Dict[str, Any]]  # 天气过滤后的景点
    daily_time_limit: int  # 每日游玩时间限制（小时）
    room_requirements: int  # 需要的房间数量
    
    # 细粒度的选择结果
    selected_restaurants: List[Dict[str, Any]]  # 选中的餐厅
    selected_hotels: List[Dict[str, Any]]  # 选中的酒店
    transportation_plan: List[Dict[str, Any]]  # 交通规划
    
    daily_route_plan: List[Dict[str, Any]]  # 每日路线规划
    time_feasible_routes: List[Dict[str, Any]]  # 时间可行的路线
    intensity_feasible_routes: List[Dict[str, Any]]  # 强度可行的路线
    budget_feasible_plan: Dict[str, Any]  # 预算可行的最终方案
    
    # 约束处理状态
    constraint_conflicts: List[str]  # 当前的约束冲突
    backtrack_history: List[str]  # 回退历史
    optimization_attempts: int  # 优化尝试次数