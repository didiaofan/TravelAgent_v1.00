# 旅行Agent项目

## 项目结构

```
旅行Agent/
├── agent_V1.py              # 原始版本1（保持不变）
├── agent_V2.py              # 原始版本2（保持不变）
├── data/                    # 数据文件夹
│   └── beijing_poi.json    # 北京景点数据
├── tools/                   # 工具文件夹
│   ├── __init__.py
│   ├── base_tool.py         # 工具基类
│   ├── weather.py           # 天气查询工具
│   ├── hotel.py             # 酒店查询工具
│   └── restaurant.py        # 餐厅查询工具
├── src/                     # 重构后的核心代码
│   ├── __init__.py
│   ├── models.py            # 数据模型
│   ├── llm_utils.py         # LLM相关工具
│   ├── poi_utils.py         # 景点相关工具
│   ├── workflow.py          # 工作流逻辑
│   └── main.py              # 主入口
├── config.py                # 配置文件
├── requirements.txt          # 依赖包
├── env_example.txt          # 环境变量模板
└── README.md                # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

1. 复制 `env_example.txt` 为 `.env`
2. 在 `.env` 文件中填入你的API密钥

```bash
# 复制环境变量模板
copy env_example.txt .env

# 编辑 .env 文件，填入你的API密钥
OPENAI_API_KEY=your_openai_api_key_here
WOKA_API_BASE=https://4.0.wokaai.com/v1
WOKA_MODEL_NAME=gpt-3.5-turbo
```

## 使用方法

### 使用重构后的代码

```python
from src import run_travel_agent_multi_turn

# 运行多轮对话
user_input = "我带孩子从上海到北京玩两天"
result = run_travel_agent_multi_turn(user_input)
print(result)
```

### 使用原始代码

```python
# 直接运行原始文件
python agent_V1.py
python agent_V2.py
```

## 重构说明

- **保持兼容性**: `agent_V1.py` 和 `agent_V2.py` 完全不变
- **模块化设计**: 按功能分离代码，便于维护和扩展
- **工具化**: 预留了天气、酒店、餐厅等查询工具的接口
- **配置集中**: 所有API密钥统一在 `.env` 文件中管理

## 扩展工具

在 `tools/` 文件夹中添加新的工具类，继承 `BaseTool` 基类：

```python
from tools.base_tool import BaseTool

class MyTool(BaseTool):
    def execute(self, **kwargs):
        # 实现你的工具逻辑
        pass
```

## 注意事项

- 确保 `.env` 文件中的API密钥正确
- 景点数据文件路径已更新为 `data/beijing_poi.json`
- 重构后的代码保持了原有的所有功能

