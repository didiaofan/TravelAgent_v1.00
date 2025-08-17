import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class Config:
    """配置管理类"""
    
    # OpenAI API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 沃卡平台配置
    WOKA_API_BASE = os.getenv("WOKA_API_BASE", "https://4.0.wokaai.com/v1")
    WOKA_MODEL_NAME = os.getenv("WOKA_MODEL_NAME", "gpt-3.5-turbo")
    
    # 其他API密钥
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    
    # 和风天气API配置
    HEFENG_API_HOST = os.getenv("HEFENG_API_HOST")
    HEFENG_API_KEY = os.getenv("HEFENG_API_KEY")
    
    HOTEL_API_KEY = os.getenv("HOTEL_API_KEY")
    TRANSPORT_API_KEY = os.getenv("TRANSPORT_API_KEY")
    
    @classmethod
    def validate(cls):
        """验证必要的配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置")
        return True

# 全局配置实例
config = Config()
