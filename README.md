# 旅行规划Agent

一个基于AI的智能旅行规划系统，能够根据用户偏好、预算、团队构成和天气情况，自动生成个性化的旅行行程。

## 🌟 主要特性

- **智能景点推荐**: 基于用户偏好和景点评分自动筛选景点
- **天气感知规划**: 根据天气预报调整行程，优先安排室内/室外景点
- **预算优化**: 智能分配预算，平衡景点门票、交通和住宿费用
- **团队适配**: 根据团队构成（成人/儿童/老人）调整行程强度
- **地理聚类**: 基于地理位置优化每日行程，减少交通时间
- **多轮对话**: 支持自然语言交互，逐步收集用户需求

## 🏗️ 项目结构

```
旅行Agent/
├── src/                          # 核心源代码
│   ├── main.py                   # 主程序入口
│   ├── workflow.py               # 工作流定义和状态管理
│   ├── poi_utils.py              # 景点数据处理和筛选
│   ├── weather_classifier.py     # 天气分类和适宜性分析
│   ├── improved_clustering.py    # 智能景点聚类算法
│   ├── models.py                 # 数据模型定义
│   └── llm_utils.py             # LLM工具和提示词管理
├── tools/                        # 外部工具集成
│   ├── hotel.py                  # 酒店搜索和预订
│   ├── weather.py                # 天气API集成
│   └── routeinf.py               # 交通路线查询
├── data/                         # 数据文件
│   └── beijing_poi.json         # 北京景点数据库
├── tests/                        # 测试文件
└── config.py                     # 配置文件
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 必要的API密钥（和风天气、交通等）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
# 单次执行
python src/main.py

# 或者作为模块导入
from src.main import run_travel_agent_multi_turn

result = run_travel_agent_multi_turn("我带孩子从上海到北京玩两天，时间是2025-08-23至2025-08-25，想去故宫和环球影城，预算2800元")
```

## 💡 使用示例

### 基本用法

```python
from src.main import run_travel_agent_multi_turn

# 用户输入示例
user_input = """
我带孩子从上海到北京玩两天
时间是2025-08-23至2025-08-25
想去故宫和环球影城
只有我和孩子2个人
两天预算2800元
"""

# 运行旅行规划
result = run_travel_agent_multi_turn(user_input)
print(result)
```

### 支持的查询类型

- **目的地**: 目前支持北京地区
- **时间**: YYYY-MM-DD格式
- **团队**: 成人、儿童、老人数量
- **预算**: 总预算或每日预算
- **偏好**: 景点类型、必去景点、避免项目
- **约束**: 酒店要求、交通偏好

## 🔧 核心功能详解

### 1. 景点筛选系统

- **偏好匹配**: 根据用户兴趣和必去景点自动筛选
- **团队适配**: 考虑儿童和老人的特殊需求
- **评分排序**: 综合景点受欢迎度和用户偏好

### 2. 天气智能规划

- **天气分类**: 将天气分为户外适宜、室内适宜、不建议出行
- **动态调整**: 根据天气预报调整景点选择
- **室内外平衡**: 雨天优先安排室内景点

### 3. 行程优化算法

- **地理聚类**: 基于地理位置优化每日行程
- **时间预算**: 智能分配每日游玩时间
- **交通优化**: 考虑景点间交通时间和费用

### 4. 预算管理

- **成本计算**: 自动计算门票、交通、住宿费用
- **预算分配**: 在预算约束下优化行程
- **费用对比**: 提供多种方案的费用分析

## 📊 数据模型

### 主要数据结构

```python
class AgentState:
    structured_info: Dict[str, Any]      # 结构化信息
    conversation: List[Dict[str, str]]    # 对话历史
    candidate_pois: List[Dict]           # 候选景点
    daily_candidates: List[Dict]         # 每日行程
    selected_hotels: List[Dict]          # 选中的酒店
    transportation_plan: List[Dict]      # 交通规划
    calculated_cost: float               # 计算出的总成本
    calculated_intensity: float          # 行程强度
```

### 景点数据结构

```python
{
    "name": "故宫博物院",
    "suggested_duration_hours": 3.0,
    "ticket_price": 60,
    "popularity_score": 9.5,
    "tags": ["历史", "文化", "博物馆"],
    "suitable_for": ["家庭", "儿童", "老人"],
    "indoor": True,
    "location": {"lat": 39.9163, "lng": 116.3972}
}
```

## 🔌 API集成

### 天气API (和风天气)

- 7天天气预报
- 天气适宜性分析
- 出行建议生成

### 交通API

- 公交路线查询
- 出租车费用估算
- 实时交通信息

### 酒店API

- 携程酒店搜索
- 价格和评分信息
- 位置和设施详情

## 🧪 测试

项目包含完整的测试套件：

```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定测试
python tests/test_weather_filter.py
python tests/test_budget_calculation.py
```

## 📝 配置说明

在 `config.py` 中配置必要的API密钥：

```python
class config:
    WOKA_API_BASE = "your_woka_api_base"
    WOKA_MODEL_NAME = "your_model_name"
    OPENAI_API_KEY = "your_openai_api_key"
    TRANSPORT_API_KEY = "your_transport_api_key"
    WEATHER_API_KEY = "your_weather_api_key"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 常见问题

### Q: 如何添加新的目的地？
A: 在 `data/` 目录下添加新的POI数据文件，并更新相关的地理信息。

### Q: 如何自定义天气分类规则？
A: 修改 `src/weather_classifier.py` 中的天气分类逻辑。

### Q: 如何集成新的交通API？
A: 在 `tools/routeinf.py` 中添加新的API实现。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 本项目仍在积极开发中，功能可能会有所变化。建议定期查看更新日志。