from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具功能"""
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数"""
        return True
    
    def get_description(self) -> str:
        """获取工具描述"""
        return f"{self.name} - 旅行规划工具"
