#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 preference_filter 节点
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poi_utils import generate_preference_filtered_candidates

def test_preference_filter_node():
    """测试 preference_filter 节点的输出"""
    
    print("=== 测试 preference_filter 节点 ===\n")
    
    # 测试用例：带孩子的家庭旅行
    print("📋 测试用例：带孩子的北京2天游")
    
    group = {
        "adults": 1,
        "children": 1,
        "elderly": 0
    }
    
    preferences = {
        "attraction_types": ["主题乐园", "博物馆", "历史"],
        "must_visit": ["故宫博物院", "北京环球影城"],
        "cuisine": ["川菜"],
        "avoid": ["酒吧", "夜店"]
    }
    
    trip_days = 2
    
    print(f"游玩天数：{trip_days}天")
    print(f"团队构成：{group}")
    print(f"偏好类型：{preferences['attraction_types']}")
    print(f"必去景点：{preferences['must_visit']}")
    print(f"避免景点：{preferences['avoid']}")
    print(f"预期最少候选数：{trip_days * 4}个")
    
    print("\n" + "="*60)
    print("🚀 执行 preference_filter 核心逻辑...")
    print("="*60)
    
    # 调用核心函数
    candidates = generate_preference_filtered_candidates(group, preferences, trip_days)
    
    print("\n" + "="*60)
    print("📊 执行结果：")
    print("="*60)
    
    if candidates:
        print(f"✅ 成功生成 {len(candidates)} 个候选景点：\n")
        
        # 检查必去景点是否包含
        must_visit = set(preferences['must_visit'])
        found_must_visit = []
        
        print("🏆 候选景点列表（按得分排序）：")
        print("-" * 60)
        for i, poi in enumerate(candidates, 1):
            score = poi.get('computed_score', 0)
            name = poi.get('name', '未知')
            tags = ', '.join(poi.get('tags', []))
            ticket_price = poi.get('ticket_price', 0)
            duration = poi.get('suggested_duration_hours', 0)
            
            # 标记必去景点
            mark = ""
            if name in must_visit:
                mark = " ⭐ [必去]"
                found_must_visit.append(name)
            
            print(f"{i:2d}. {name}{mark}")
            print(f"    得分: {score:.3f} | 门票: {ticket_price}元 | 时长: {duration}h")
            print(f"    标签: {tags}")
            print()
        
        print(f"🎯 必去景点包含情况：{len(found_must_visit)}/{len(must_visit)}")
        for name in must_visit:
            status = "✅" if name in found_must_visit else "❌"
            print(f"  {status} {name}")
        
        print(f"\n📈 得分分布：")
        scores = [poi.get('computed_score', 0) for poi in candidates]
        print(f"  最高分: {max(scores):.3f}")
        print(f"  最低分: {min(scores):.3f}")
        print(f"  平均分: {sum(scores)/len(scores):.3f}")
        
        # 验证结果合理性
        print(f"\n✅ 测试结果验证：")
        print(f"  候选数量满足要求: {len(candidates) >= trip_days * 4}")
        print(f"  包含必去景点: {len(found_must_visit) > 0}")
        print(f"  得分排序正确: {all(candidates[i]['computed_score'] >= candidates[i+1]['computed_score'] for i in range(len(candidates)-1))}")
        
    else:
        print("❌ 未生成任何候选景点")
        print("💡 可能原因：")
        print("  - POI数据文件读取失败")
        print("  - 过滤条件过于严格")
        print("  - 团队构成不匹配任何景点")
    
    return candidates

def test_multiple_scenarios():
    """测试多种场景"""
    
    print("\n" + "="*80)
    print("🧪 多场景测试")
    print("="*80)
    
    scenarios = [
        {
            "name": "成人文化之旅",
            "group": {"adults": 2, "children": 0, "elderly": 0},
            "preferences": {
                "attraction_types": ["历史", "博物馆", "文化遗产"],
                "must_visit": ["天安门广场", "天坛"],
                "cuisine": ["北京菜"],
                "avoid": ["主题乐园"]
            },
            "trip_days": 3
        },
        {
            "name": "老年人休闲游",
            "group": {"adults": 1, "children": 0, "elderly": 2},
            "preferences": {
                "attraction_types": ["皇家园林", "公园"],
                "must_visit": ["颐和园"],
                "cuisine": [],
                "avoid": ["刺激", "高强度"]
            },
            "trip_days": 1
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        print(f"  团队: {scenario['group']}")
        print(f"  天数: {scenario['trip_days']}天")
        print(f"  必去: {scenario['preferences']['must_visit']}")
        
        candidates = generate_preference_filtered_candidates(
            scenario['group'], 
            scenario['preferences'], 
            scenario['trip_days']
        )
        
        print(f"  结果: {len(candidates)} 个候选景点")
        if candidates:
            top3 = candidates[:3]
            for i, poi in enumerate(top3, 1):
                print(f"    {i}. {poi['name']} (得分: {poi['computed_score']:.3f})")

if __name__ == "__main__":
    # 主要测试
    result = test_preference_filter_node()
    
    # 多场景测试
    test_multiple_scenarios()
    
    print("\n🎉 测试完成！")
