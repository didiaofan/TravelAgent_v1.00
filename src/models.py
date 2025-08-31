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
    
    # 新状态图的数据结构
    daily_candidates: List[Dict[str, Any]]  # 每日候选景点列表
    selected_restaurants: List[Dict[str, Any]]  # 选中的餐厅
    selected_hotels: List[Dict[str, Any]]  # 选中的酒店
    transportation_plan: List[Dict[str, Any]]  # 交通规划
    
    # 强度相关
    calculated_intensity: float  # 计算出的强度值
    intensity_satisfied: bool  # 强度是否满足
    intensity_optimization_attempts: int  # 强度优化尝试次数
    can_optimize_intensity: bool  # 是否可以优化强度
    valid_transport_plans: List[Dict[str, Any]]  # 符合强度约束的交通方案
    
    # 预算相关
    calculated_cost: float  # 计算出的总成本
    cost_breakdown: Dict[str, Any]  # 成本分解
    budget_satisfied: bool  # 预算是否满足
    budget_optimization_target: str  # 预算优化目标
    recommended_plan: Dict[str, Any]  # 推荐的最优方案
    all_plan_costs: List[Dict[str, Any]]  # 所有方案的费用对比
    budget_check_result: str  # 预算检查结果
    
    # 优化控制标记
    hotel_optimization_blocked: bool  # 酒店优化是否被阻塞
    transport_optimization_blocked: bool  # 交通优化是否被阻塞
    restaurant_optimization_blocked: bool  # 餐厅优化是否被阻塞
    is_optimization_round: bool  # 是否为优化轮次
    
    # 优化后的数据
    optimized_hotels: List[Dict[str, Any]]  # 优化后的酒店
    optimized_transportation_plan: List[Dict[str, Any]]  # 优化后的交通计划
    optimized_restaurants: List[Dict[str, Any]]  # 优化后的餐厅
    
    # 每日景点数据
    daily_available_pois: List[Dict[str, Any]]  # 每日可访问景点详细信息
    
    # 酒店搜索数据
    hotel_search_results: List[Dict[str, Any]]  # 酒店搜索结果
    hotel_selection_history: List[Dict[str, Any]]  # 酒店选择历史
    hotel_optimization_attempts: int  # 酒店优化尝试次数
    max_hotel_optimization_attempts: int  # 最大酒店优化次数
    excluded_hotels: List[str]  # 被排除的酒店名称列表
    
    # 交通规划数据
    transportation_plans: Dict[str, Any]  # 交通规划方案
    
    # 强度计算数据
    intensity_calculation_result: Dict[str, Any]  # 强度计算结果