#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气分类器
根据和风天气API返回的textDay信息对天气进行分类，并评估出行适宜性
"""

from typing import Dict, List, Tuple
from enum import Enum

class WeatherSuitability(Enum):
    """出行适宜性等级"""
    OUTDOOR_SUITABLE = "可户外出行"      # 适合户外景点
    INDOOR_SUITABLE = "可市内出行"       # 只适合室内景点  
    NOT_RECOMMENDED = "不建议出行"       # 不建议出行

class WeatherClassifier:
    """天气分类器"""
    
    def __init__(self):
        # 基于和风天气textDay描述的天气分类
        # 参考: https://dev.qweather.com/docs/api/weather/weather-daily-forecast/#request-example
        # 参考: https://icons.qweather.com/
        
        self.weather_categories = {
            # 可户外出行 - 晴好天气
            WeatherSuitability.OUTDOOR_SUITABLE: [
                "晴", "多云", "少云", "晴间多云", "阴", 
                "薄雾", "霾", "浮尘", "扬沙", "沙尘",
                "雾", "冰雹", "雨夹雪" ,"雷阵雨" # 轻微不良天气但仍可出行
            ],
            
            # 可市内出行 - 降水天气（适合室内景点）
            WeatherSuitability.INDOOR_SUITABLE: [
                "阵雨",  "雷阵雨伴有冰雹", "小雨", "中雨", 
                 "冻雨", "雨雪天气", "阵雨转多云", "小到中雨", 
                "小雪", "中雪",  "雨夹雪",
                "阵雪", "小到中雪",
            ],
            
            # 不建议出行 - 极端天气
            WeatherSuitability.NOT_RECOMMENDED: [
                "大雨", "暴雨", "大暴雨", "特大暴雨",
                "中到大雨", "大雨到暴雨", "暴雨到大暴雨", "大暴雨到特大暴雨",
                "大雪", "暴雪","中到大雪", "大到暴雪"
                "强沙尘暴", "龙卷风", "大风", "烈风", "狂风", 
                "飓风", "热带风暴", "强热带风暴", "台风", "强台风", "超强台风",
                "特强沙尘暴", "沙尘暴", "强对流天气", "雷暴",
                "极端高温", "极端低温", "冰冻", "严重冰冻"
            ]
        }
        
        # 创建反向查找字典
        self.text_to_suitability = {}
        for suitability, weather_list in self.weather_categories.items():
            for weather in weather_list:
                self.text_to_suitability[weather] = suitability
    
    def classify_weather(self, text_day: str) -> WeatherSuitability:
        """
        根据textDay分类天气适宜性
        
        Args:
            text_day: 和风天气API返回的白天天气描述
            
        Returns:
            天气适宜性等级
        """
        # 精确匹配
        if text_day in self.text_to_suitability:
            return self.text_to_suitability[text_day]
        
        # 模糊匹配 - 检查关键词
        text_lower = text_day.lower()
        
        # 检查极端天气关键词
        extreme_keywords = ["暴", "狂", "龙卷", "台风", "飓风", "冰冻", "极端"]
        if any(keyword in text_day for keyword in extreme_keywords):
            return WeatherSuitability.NOT_RECOMMENDED
        
        # 检查降水关键词  
        rain_keywords = ["雨", "雪", "雷", "冰雹"]
        if any(keyword in text_day for keyword in rain_keywords):
            return WeatherSuitability.INDOOR_SUITABLE
        
        # 检查大风关键词
        wind_keywords = ["大风", "强风", "烈风", "沙尘暴"]
        if any(keyword in text_day for keyword in wind_keywords):
            return WeatherSuitability.NOT_RECOMMENDED
        
        # 默认为可户外出行
        return WeatherSuitability.OUTDOOR_SUITABLE
    
    def get_suitable_pois(self, candidate_pois: List[Dict], weather_suitability: WeatherSuitability) -> List[Dict]:
        """
        根据天气适宜性筛选景点
        
        Args:
            candidate_pois: 候选景点列表
            weather_suitability: 天气适宜性等级
            
        Returns:
            筛选后的景点列表
        """
        if weather_suitability == WeatherSuitability.OUTDOOR_SUITABLE:
            # 晴好天气：所有景点都适合
            return candidate_pois
        
        elif weather_suitability == WeatherSuitability.INDOOR_SUITABLE:
            # 降水天气：只选择室内景点
            indoor_pois = []
            for poi in candidate_pois:
                indoor_status = poi.get("indoor")
                # indoor字段为True表示室内，或者包含"室内"、"博物馆"、"馆"等关键词
                if (indoor_status is True or 
                    (isinstance(indoor_status, str) and "室内" in indoor_status) or
                    any(keyword in poi.get("name", "") for keyword in ["博物馆", "美术馆", "科技馆", "展览馆", "商场", "购物中心"])):
                    indoor_pois.append(poi)
            return indoor_pois
        
        elif weather_suitability == WeatherSuitability.NOT_RECOMMENDED:
            # 极端天气：不推荐任何景点
            return []
        
        return candidate_pois
    
    def analyze_trip_weather(self, weather_data: List[Dict], trip_dates: List[str]) -> Dict:
        """
        分析行程期间的天气情况
        
        Args:
            weather_data: 和风天气API返回的每日天气数据
            trip_dates: 行程日期列表 (格式: YYYY-MM-DD)
            
        Returns:
            天气分析结果
        """
        weather_analysis = {
            "daily_weather": [],
            "overall_assessment": "",
            "recommendations": [],
            "extreme_weather_days": 0,
            "suitable_days": 0,
            "indoor_days": 0
        }
        
        suitable_days = 0
        indoor_days = 0
        bad_weather_days = 0
        
        # 创建日期到天气的映射
        weather_by_date = {item["fxDate"]: item for item in weather_data}
        
        for date in trip_dates:
            if date in weather_by_date:
                day_weather = weather_by_date[date]
                text_day = day_weather.get("textDay", "未知")
                suitability = self.classify_weather(text_day)
                
                day_info = {
                    "date": date,
                    "text_day": text_day,
                    "temp_max": day_weather.get("tempMax"),
                    "temp_min": day_weather.get("tempMin"),
                    "precip": day_weather.get("precip", "0.0"),
                    "suitability": suitability,
                    "suitability_text": suitability.value
                }
                
                weather_analysis["daily_weather"].append(day_info)
                
                # 统计天气情况
                if suitability == WeatherSuitability.OUTDOOR_SUITABLE:
                    suitable_days += 1
                elif suitability == WeatherSuitability.INDOOR_SUITABLE:
                    indoor_days += 1
                else:
                    bad_weather_days += 1
        
        # 保存统计数据
        weather_analysis["extreme_weather_days"] = bad_weather_days
        weather_analysis["suitable_days"] = suitable_days
        weather_analysis["indoor_days"] = indoor_days
        
        # 生成总体评估
        total_days = len(trip_dates)
        if bad_weather_days > 0:
            weather_analysis["overall_assessment"] = f"行程期间有{bad_weather_days}天极端天气，需要特别注意"
        elif indoor_days > total_days // 2:
            weather_analysis["overall_assessment"] = f"行程期间多为降水天气({indoor_days}天)，建议以室内景点为主"
        elif suitable_days > total_days // 2:
            weather_analysis["overall_assessment"] = f"行程期间天气良好({suitable_days}天)，适合户外活动"
        else:
            weather_analysis["overall_assessment"] = "行程期间天气多变，建议灵活安排"
        
        # 生成建议
        if bad_weather_days > 0:
            weather_analysis["recommendations"].append("考虑调整行程日期，避开极端天气")
        if indoor_days > 0:
            weather_analysis["recommendations"].append("准备雨具，优先安排室内景点")
        if suitable_days > 0:
            weather_analysis["recommendations"].append("充分利用好天气安排户外景点")
        
        return weather_analysis
    
    def check_extreme_weather_blocking(self, weather_analysis: Dict, total_trip_days: int) -> bool:
        """
        检查是否有极端天气导致不能满足约定的出行天数
        
        Args:
            weather_analysis: 天气分析结果
            total_trip_days: 计划的出行天数
            
        Returns:
            True: 极端天气阻断出行, False: 可以正常出行
        """
        extreme_days = weather_analysis.get("extreme_weather_days", 0)
        
        # 如果极端天气天数 >= 总天数，则认为无法满足出行要求
        return extreme_days >= total_trip_days
    
    def check_must_visit_weather_conflict(self, weather_analysis: Dict, must_visit_pois: List[Dict]) -> bool:
        """
        检查必去景点是否受天气影响无法访问
        
        Args:
            weather_analysis: 天气分析结果
            must_visit_pois: 必去景点列表
            
        Returns:
            True: 必去景点受天气影响无法访问, False: 必去景点可以正常访问
        """
        if not must_visit_pois:
            return False
            
        # 统计各种天气的天数
        extreme_days = weather_analysis.get("extreme_weather_days", 0)
        indoor_days = weather_analysis.get("indoor_days", 0)
        suitable_days = weather_analysis.get("suitable_days", 0)
        total_days = len(weather_analysis.get("daily_weather", []))
        
        # 检查必去景点的室内外属性
        outdoor_must_visit = []
        for poi in must_visit_pois:
            indoor_status = poi.get("indoor")
            if indoor_status is False or (isinstance(indoor_status, str) and "室内" not in indoor_status and 
                                         not any(keyword in poi.get("name", "") for keyword in ["博物馆", "美术馆", "科技馆", "展览馆", "商场", "购物中心"])):
                outdoor_must_visit.append(poi)
        
        # 如果有户外必去景点，但所有天气都不适合户外活动
        if outdoor_must_visit and (extreme_days + indoor_days) == total_days:
            return True
            
        return False
    
    def filter_completely_inaccessible_pois(self, candidate_pois: List[Dict], weather_analysis: Dict) -> List[Dict]:
        """
        根据天气情况筛选景点，剔除在整个行程期间完全不能到达的景点
        
        Args:
            candidate_pois: 候选景点列表
            weather_analysis: 天气分析结果
            
        Returns:
            筛选后的景点列表
        """
        # 统计各种天气的天数
        extreme_days = weather_analysis.get("extreme_weather_days", 0)
        indoor_days = weather_analysis.get("indoor_days", 0)
        suitable_days = weather_analysis.get("suitable_days", 0)
        total_days = len(weather_analysis.get("daily_weather", []))
        
        filtered_pois = []
        
        for poi in candidate_pois:
            indoor_status = poi.get("indoor")
            
            # 判断是否为混合型景点
            is_mixed = (isinstance(indoor_status, str) and "混合" in indoor_status)
            
            # 判断是否为纯室内景点
            is_pure_indoor = (indoor_status is True or 
                            (isinstance(indoor_status, str) and "室内" in indoor_status and "混合" not in indoor_status) or
                            any(keyword in poi.get("name", "") for keyword in ["博物馆", "美术馆", "科技馆", "展览馆", "商场", "购物中心"]))
            
            # 判断是否为纯户外景点
            is_pure_outdoor = (indoor_status is False or 
                             (isinstance(indoor_status, str) and "室内" not in indoor_status and "混合" not in indoor_status))
            
            # 筛选逻辑：
            # 1. 如果所有天气都是极端天气，不保留任何景点
            if extreme_days == total_days:
                continue
                
            # 2. 如果所有天气都不适合户外，只保留室内和混合景点  
            elif (extreme_days + indoor_days) == total_days and is_pure_outdoor:
                continue
                
            # 3. 其他情况保留景点（好天气时保留所有景点，用户可自由选择）
            else:
                filtered_pois.append(poi)
        
        return filtered_pois
    
    def check_trip_fullness(self, filtered_pois: List[Dict], daily_time_budget: float, trip_days: int) -> Tuple[bool, Dict]:
        """
        检查行程是否饱满
        
        Args:
            filtered_pois: 筛选后的景点列表
            daily_time_budget: 团队每日可游玩时间（小时）
            trip_days: 游玩天数
            
        Returns:
            (is_full, analysis): (是否饱满, 分析结果)
        """
        # 计算行程时间
        total_time_budget = daily_time_budget * trip_days
        
        # 计算剩余景点的建议游玩总时间
        total_suggested_hours = sum(poi.get("suggested_duration_hours", 2.0) for poi in filtered_pois)
        
        # 计算时间差
        time_difference = total_time_budget - total_suggested_hours
        
        # 判断是否饱满：时间差 > 10小时代表景点太少
        is_full = time_difference <= 10
        
        analysis = {
            "total_time_budget": total_time_budget,
            "total_suggested_hours": total_suggested_hours,
            "time_difference": time_difference,
            "is_full": is_full,
            "fullness_percentage": (total_suggested_hours / total_time_budget * 100) if total_time_budget > 0 else 0
        }
        
        return is_full, analysis

def format_weather_analysis(weather_analysis: Dict) -> str:
    """
    格式化天气分析结果为易读文本
    
    Args:
        weather_analysis: 天气分析结果
        
    Returns:
        格式化的文本
    """
    lines = []
    lines.append("🌤️ 行程天气分析")
    lines.append("=" * 50)
    
    # 每日天气详情
    lines.append("\n📅 每日天气详情:")
    for day_info in weather_analysis["daily_weather"]:
        temp_range = f"{day_info['temp_min']}°C ~ {day_info['temp_max']}°C"
        precip = f"降水: {day_info['precip']}mm" if float(day_info['precip']) > 0 else "无降水"
        
        # 适宜性图标
        suitability = day_info['suitability']
        if suitability == WeatherSuitability.OUTDOOR_SUITABLE:
            icon = "☀️"
        elif suitability == WeatherSuitability.INDOOR_SUITABLE:
            icon = "🌧️"
        else:
            icon = "⚠️"
        
        lines.append(f"{icon} {day_info['date']}: {day_info['text_day']}")
        lines.append(f"   温度: {temp_range} | {precip} | {day_info['suitability_text']}")
    
    # 总体评估
    lines.append(f"\n📊 总体评估: {weather_analysis['overall_assessment']}")
    
    # 建议
    if weather_analysis['recommendations']:
        lines.append("\n💡 出行建议:")
        for i, rec in enumerate(weather_analysis['recommendations'], 1):
            lines.append(f"  {i}. {rec}")
    
    return "\n".join(lines)
