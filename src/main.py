import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow import create_agent_workflow, init_state, MAX_CONVERSATION_STEPS
from langgraph.graph import END

# 主函数：运行Agent（单次执行）
def run_travel_agent(user_input: str):
    """单次执行旅行Agent"""
    # 初始化状态
    state = init_state(user_input)
    workflow = create_agent_workflow()
    
    # 运行工作流
    try:
        # 执行工作流
        result = workflow.invoke(state)
        
        # 检查是否结束
        if END in result:
            return result[END]["structured_info"]
        
        # 如果没有结束，说明需要用户输入
        return result["structured_info"]
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("将使用当前信息生成行程...")
        return state["structured_info"]

# 多轮对话函数（避免递归）
def run_travel_agent_multi_turn(initial_input: str, max_turns: int = 5):
    """多轮对话版本，避免递归问题"""
    # 初始化状态
    state = init_state(initial_input)
    workflow = create_agent_workflow()
    
    turn_count = 0
    
    while turn_count < max_turns:
        try:
            print(f"\n=== 第 {turn_count + 1} 轮对话 ===")
            
            # 执行工作流
            result = workflow.invoke(state)
            
            # 如果已无缺失字段或达到最大轮次，直接结束
            if (not result.get('missing_fields')) or (result.get('step_count', 0) >= MAX_CONVERSATION_STEPS):
                print("信息收集完成！")
                return result.get('structured_info', state['structured_info'])
            
            # 显示当前状态与缺失字段
            print(f"当前已收集信息: {json.dumps(result['structured_info'], ensure_ascii=False, indent=2)}")
            print(f"缺失字段: {result['missing_fields']}")
            
            # 仅当存在缺失字段时显示助手追问
            if result['missing_fields']:
                last_assistant_msg = None
                for msg in reversed(result.get('conversation', [])):
                    if msg.get('role') == 'assistant':
                        last_assistant_msg = msg
                        break
                if last_assistant_msg:
                    print(f"Assistant: {last_assistant_msg['content']}")
            
            # 获取用户输入（仅在缺失字段时）
            user_response = input("User: ")
            if user_response.lower() in ['quit', 'exit', '结束', '退出']:
                print("用户选择退出，使用当前信息生成行程。")
                return result['structured_info']
            
            # 更新状态，准备下一轮
            result['conversation'].append({"role": "user", "content": user_response})
            state = result
            turn_count += 1
            
        except Exception as e:
            print(f"发生错误: {e}")
            print("将使用当前信息生成行程...")
            return state["structured_info"]
    
    print("达到最大对话轮次，使用当前信息生成行程。")
    return state["structured_info"]



# 主程序入口
if __name__ == "__main__":
    user_input = "我带孩子从上海到北京玩两天，时间是2025-08-23至2025-08-25，想去故宫和环球影城和北京野生动物园，只有我和孩子2个人，两天预8000"
    
    print("=== 旅行规划Agent V4 ===")
    print(f"User: {user_input}")
    
    try:
        # 使用多轮对话版本，避免递归问题
        final_info = run_travel_agent_multi_turn(user_input, max_turns=5)
        
        # itinerary_text = final_info.get('itinerary_text') if isinstance(final_info, dict) else None
        # if itinerary_text:
        #     print("\n=== 行程方案 ===")
        #     print(itinerary_text)
        #     total_cost = final_info.get('total_cost')
        #     if total_cost is not None:
        #         print(f"\n总花费：{total_cost} 元")
        # else:
        #     print("\n=== 结构化输出 ===")
        #     print(json.dumps(final_info, ensure_ascii=False, indent=2))
            
    except KeyboardInterrupt:
        print("\n👋 程序退出")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        print("💡 如需测试单个节点，请使用 tests/ 目录下的测试文件")
