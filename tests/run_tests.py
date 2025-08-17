#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行器 - 统一运行所有节点测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """运行所有可用的测试"""
    
    print("🧪 旅行Agent节点测试套件")
    print("="*50)
    
    # 测试列表（按开发顺序）
    tests = [
        ("preference_filter", "偏好筛选节点"),
        ("qweather", "和风天气API测试"),
        ("weather_filter", "天气过滤节点"),
        # 后续可以添加更多测试：
        # ("team_constraints", "团队约束节点"),
        # ("restaurant_selection", "餐厅选择节点"),
        # ("hotel_selection", "酒店选择节点"),
        # ("transportation_planning", "交通规划节点"),
        # ("route_optimization", "路线优化节点"),
        # ("time_window_check", "时间窗口检查节点"),
        # ("intensity_check", "强度检查节点"),
        # ("budget_check", "预算检查节点"),
    ]
    
    print(f"📋 共有 {len(tests)} 个测试可运行\n")
    
    for i, (test_name, description) in enumerate(tests, 1):
        print(f"{i}. {description} (test_{test_name}.py)")
    
    print("\n选择要运行的测试：")
    print("0. 运行所有测试")
    
    try:
        choice = input("请输入选择 (0-{}): ".format(len(tests)))
        choice = int(choice)
        
        if choice == 0:
            # 运行所有测试
            for test_name, description in tests:
                print(f"\n{'='*60}")
                print(f"🧪 运行测试: {description}")
                print(f"{'='*60}")
                run_single_test(test_name)
        elif 1 <= choice <= len(tests):
            # 运行单个测试
            test_name, description = tests[choice - 1]
            print(f"\n{'='*60}")
            print(f"🧪 运行测试: {description}")
            print(f"{'='*60}")
            run_single_test(test_name)
        else:
            print("❌ 无效选择")
            
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 测试中断")

def run_single_test(test_name):
    """运行单个测试"""
    try:
        if test_name == "preference_filter":
            from test_preference_filter import test_preference_filter_node
            test_preference_filter_node()
        elif test_name == "qweather":
            print("⚠️ 和风天气API测试已移除，请使用weather_filter测试")
        elif test_name == "weather_filter":
            from test_weather_filter import test_weather_classifier, test_poi_filtering, test_trip_weather_analysis
            test_weather_classifier()
            test_poi_filtering() 
            test_trip_weather_analysis()
        # 后续可以添加更多测试的导入和调用
        else:
            print(f"❌ 测试 {test_name} 尚未实现")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    run_all_tests()
