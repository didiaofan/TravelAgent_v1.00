#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试POI数据加载问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poi_utils import load_poi_data

def debug_poi_loading():
    """调试POI数据加载"""
    
    print("🔍 调试POI数据加载...")
    
    # 测试不同的路径
    paths_to_try = [
        "data/beijing_poi.json",
        "../data/beijing_poi.json", 
        os.path.join(os.path.dirname(__file__), '..', 'data', 'beijing_poi.json'),
        os.path.join(os.getcwd(), 'data', 'beijing_poi.json')
    ]
    
    for i, path in enumerate(paths_to_try, 1):
        print(f"\n{i}. 尝试路径: {path}")
        print(f"   绝对路径: {os.path.abspath(path)}")
        print(f"   文件存在: {os.path.exists(path)}")
        
        if os.path.exists(path):
            try:
                data = load_poi_data(path)
                print(f"   ✅ 成功加载 {len(data)} 个景点")
                if data:
                    print(f"   第一个景点: {data[0].get('name', '未知')}")
                break
            except Exception as e:
                print(f"   ❌ 加载失败: {str(e)}")
        else:
            print(f"   ❌ 文件不存在")
    
    print(f"\n当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {os.path.dirname(__file__)}")

if __name__ == "__main__":
    debug_poi_loading()
