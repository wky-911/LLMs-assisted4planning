import re
import math

from components.model_parser.parser_new import *

# exploration_weight = 0.8
exploration_weight = 0

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # 当前节点的状态
        self.parent = parent  # 父节点
        self.action = action  # 从父节点到达此节点的动作
        self.children = []  # 子节点列表
        self.visits = 0  # 被访问次数
        self.reward = 0  # 累计奖励

    def is_fully_expanded(self, actions):
        """检查节点是否已经完全扩展（所有可能的动作均已被尝试）。"""
        tried_actions = [(child.action.action_name, child.action.parameters) for child in self.children]
        return set(tried_actions) == set([(action.action_name, action.parameters) for action in actions])

    def best_child(self):
        """根据 UCB1 公式选择该节点下的最佳子节点。"""
        def ucb1(node):
            if node.visits == 0:
                # 如果节点的访问次数为 0，则返回一个非常大的值，确保该节点被选择
                return float('inf')  # 将探索权值设为无穷大
            exploitation = node.reward / node.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / node.visits) if exploitation > 0 else 0
            return exploitation + exploration

        print(f"max: {max(self.children, key=ucb1)}, is best child")
        return max(self.children, key=ucb1)

    def __str__(self):
        return f"前置动作：{self.action} 节点价值：{' '.join(str(self.reward))} 访问次数：{self.visits}"


class Action:
    def __init__(self, name, params):
        self.action_name = name
        self.parameters = params

    def __str__(self):
        return f"({self.action_name} {' '.join(self.parameters)})"


ai_key = os.getenv('DASHSCOPE_API_KEY')

def list_to_tuple(data):
    """循环递归地将列表元素转换为元组"""
    if isinstance(data, list):
        return tuple(list_to_tuple(item) for item in data)
    return data


def tuple_to_list(data):
    """递归地将列表内部的所有元组转换为列表"""
    if isinstance(data, list):
        return [tuple_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return list(tuple_to_list(item) for item in data)
    else:
        return data

def bind_parameters(action_effects, bindings):
    """根据 bindings 替换 action_effects 中的占位符"""
    updated_effects = []
    for condition in action_effects:
        predicate = condition[0]
        updated_vars = [bindings.get(var.strip('?'), var) for var in condition[1]]
        updated_effects.append([predicate, updated_vars])
    # print(f"updated_effects: {updated_effects}")
    return updated_effects

def apply_action(state, action, domain):
    """应用一个动作，生成新的状态。"""
    params = [p[0].replace('?', '') for p in [param for param in domain[action.action_name][PARARMETERS]]]
    action_bindings ={f"{param}": value for param, value in zip(params, action.parameters)}
    new_state = set(state)
    # print(f"new_state: {new_state}")

    # # 应用绑定
    adds = list_to_tuple(bind_parameters(domain[action.action_name][ADDS], action_bindings))  # 需要添加的状态
    dels = list_to_tuple(bind_parameters(domain[action.action_name][DELS], action_bindings))  # 需要删除的状态
    # print(f"adds: {adds}, dels: {dels}")

    for add in adds:
        new_state.add(add)
    for delete in dels:
        new_state.discard(delete)

    return set(new_state)


# def list_to_tuple(data):
#     """循环递归地将列表元素转换为元组"""
#     if isinstance(data, list):
#         return tuple(list_to_tuple(item) for item in data)
#     return data
#
#
# def tuple_to_list(data):
#     """递归地将列表内部的所有元组转换为列表"""
#     if isinstance(data, list):
#         return [tuple_to_list(item) for item in data]
#     elif isinstance(data, tuple):
#         return list(tuple_to_list(item) for item in data)
#     else:
#         return data
#
#
# def has_duplicates(input_list):
#     # 使用set来存储已经见过的元素
#     seen = set()
#     # 使用set来存储重复的元素
#     duplicates = set()
#
#     for element in input_list:
#         if element in seen:
#             duplicates.add(element)
#         else:
#             seen.add(element)
#
#     # 如果重复元素的集合非空，则表示有重复
#     return bool(duplicates)
#
#
# def find_bindings(params, conditions, state, obj):
#     # params: [('?ob', 'object')] || [('?ob', 'object'), ('?underob', 'object')]
#     # conditions: [('clear', ('?ob',)), ('ontable', ('?ob',)), ('handempty', ())] || conditions: [('clear', ('?underob',)), ('holding', ('?ob',))]
#     # print(f"params: {params}, conditions: {conditions}, state: {state}")
#     bindings = set()  # 装入所有可能的参数组合
#
#     # 一、将条件中的抽象参数替换为具体参数
#     # 二、检验该条件是否满足当前状态，如果满足则保存
#     params = [p[0].replace('?', '') for p in params] # ['ob', 'underob']
#     # print(f"after: {params}, conditions: {conditions}, replica_conditions: {[cond[1] for cond in conditions]}")
#     # 1.1如果动作涉及多个参数，则列出所有的组合
#     if len(params) > 1:
#         pairs = list(permutations(obj, 2))
#         # print("pairs: ",pairs)
#         # 1.2为抽象参数绑定上具体的参数
#         for pair in pairs:
#             replica_conditions = [tuple_to_list(cond) for cond in conditions] # [['on', ['j', 'f']], ['clear', ['j']], ['handempty', []]]
#             for cond in [condition[1] for condition in replica_conditions]:
#                 for i in range(len(cond)):
#                     if cond[i].replace('?', '') == params[0]:
#                         cond[i] = pair[0]
#                     elif cond[i].replace('?', '') == params[1]:
#                         cond[i] = pair[1]
#             # 2.1检验当前条件是否满足当前状态
#             replica_conditions = list(list_to_tuple(r_cond) for r_cond in replica_conditions)
#
#             valid = True
#             for cond in replica_conditions:
#                 if cond not in state:
#                     valid = False
#                     break
#             # print(f"valid: {valid}, replica_conditions: {replica_conditions}")
#             if valid:
#                 bindings.add(pair)
#     else:
#         pairs = [(p,) for p in obj]
#         for pair in pairs:
#             replica_conditions = [tuple_to_list(cond) for cond in conditions]
#             for cond in [condition[1] for condition in replica_conditions]:
#                 for i in range(len(cond)):
#                     cond[i] = pair[0]
#             replica_conditions = list(list_to_tuple(r_cond) for r_cond in replica_conditions)
#             valid = True
#             for cond in replica_conditions:
#                 if cond not in state:
#                     valid = False
#                     break
#             # print(f"valid: {valid}, replica_conditions: {replica_conditions}")
#             if valid:
#                 bindings.add(pair)
#
#     # print(f"pairs: {pairs}")
#     # print(f"replica_conditions: {replica_conditions}")
#     # print(f"bindings: {bindings}") # {('g',), ('b',), ('l',), ('i',), ('f',), ('h',), ('d',), ('j',)}
#     return bindings
#
#
# def applicable_actions(state, domain, obj):
#     """获取在当前状态下可以执行的动作列表。"""
#     actions = []
#     for action_name, action_info in domain.items():
#         conditions = [list_to_tuple(cond) for cond in action_info[POS_PREC]]  # 取出某个动作的执行条件
#         parameters = [param for param in action_info[PARARMETERS]]  # 取出该动作的参数
#         bindings = find_bindings(parameters, conditions, state, obj)  # 尝试为该动作找到可行的具体参数（该状态下是否可执行该动作）
#         if bindings is not None:  # 如果找到有效绑定，则该动作可执行
#             for binding in bindings:
#                 actions.append(Action(action_name, tuple(param for param in binding)))
#     random.shuffle(actions)
#     return list(actions)
#
#
# def bind_parameters(action_effects, bindings):
#     """根据 bindings 替换 action_effects 中的占位符"""
#     updated_effects = []
#     for condition in action_effects:
#         predicate = condition[0]
#         updated_vars = [bindings.get(var.strip('?'), var) for var in condition[1]]
#         updated_effects.append([predicate, updated_vars])
#     # print(f"updated_effects: {updated_effects}")
#     return updated_effects
#
#
# def apply_action(state, action, domain):
#     """应用一个动作，生成新的状态。"""
#     params = [p[0].replace('?', '') for p in [param for param in domain[action.action_name][PARARMETERS]]]
#     action_bindings ={f"{param}": value for param, value in zip(params, action.parameters)}
#     new_state = set(state)
#     # print(f"new_state: {new_state}")
#
#     # # 应用绑定
#     adds = list_to_tuple(bind_parameters(domain[action.action_name][ADDS], action_bindings))  # 需要添加的状态
#     dels = list_to_tuple(bind_parameters(domain[action.action_name][DELS], action_bindings))  # 需要删除的状态
#     # print(f"adds: {adds}, dels: {dels}")
#
#     for add in adds:
#         new_state.add(add)
#     for delete in dels:
#         new_state.discard(delete)
#
#     return set(new_state)
#
#
# def goal_achieved(current_state, goal_state):
#     """检查目标是否达成。"""
#     for goal in goal_state:
#         # 将目标状态格式化为元组
#         goal_tuple = (goal[0], tuple(goal[1]))
#         # 检查当前状态中是否存在该目标状态
#         if goal_tuple not in current_state:
#             return False
#     return True
#
#
# def back_propagate(node, reward, depth):
#     """将奖励反向传播到路径上的所有节点。"""
#     while node is not None:
#         # 让mcts尽量兼顾到短深度的节点
#         # node.visits += 1 + depth/(depth + 1)
#         node.visits += 1
#         node.reward += reward
#         node = node.parent
#
#
# def create_prompt(domain_description, init_State, current_state, goal_state, applied_actions, actions):
#     prompt = f"Here is the data I have provided for you to analyse\n"
#     prompt += f"{domain_description}\n"
#     prompt += f"Init State: {init_State}\n"
#     prompt += f"Current State: {current_state}\n"
#     prompt += f"Goal State: {goal_state}\n"
#     prompt += f"Action Trajectory: {', '.join([str((action.action_name, action.parameters)) for action in applied_actions])}\n"
#     prompt += f"Available Actions & Parameters: {', '.join([str((action.action_name, action.parameters)) for action in actions])}\n"
#
#     return prompt
#
#
# def safe_parse_llm_output(output_text):
#     # 阶段1：基础清洗
#     cleaned = re.sub(r"\s+", " ", output_text)  # 合并多余空白
#     cleaned = cleaned.replace("，", ",").replace("‘", "'").replace("’", "'")  # 统一符号
#
#     # 阶段2：修复常见语法错误
#     # 修复单元素元组缺少逗号的问题 (e.g. ('a') → ('a',))
#     cleaned = re.sub(
#         r"\(((?:'[^']+'(?:,\s*)?)+)\)",
#         lambda m: f"({m.group(1).rstrip(',')},)" if len(m.group(1).split(',')) == 1 else f"({m.group(1)})",
#         cleaned
#     )
#
#     # 新增阶段2.5：修复三元组结构错误
#     # 将 (action, params, reward) 转换为 ((action, params), reward)
#     cleaned = re.sub(
#         r"\(\s*'([^']+)'\s*,\s*(\([^)]+\)|'[^']+')\s*,\s*(\d+)\s*\)",  # 匹配错误的三元组
#         r"((\1, \2), \3)",  # 转换为正确的嵌套结构
#         cleaned
#     )
#
#     # 阶段3：分层次解析
#     try:
#         # 先尝试直接解析完整结构
#         return ast.literal_eval(cleaned)
#     except:
#         # 阶段4：容错解析
#         action_entries = re.findall(
#             r"(\(\(.*?\)\s*,\s*\d+\))",  # 匹配每个动作条目
#             cleaned.replace("'", '"')  # 统一引号便于解析
#         )
#
#         parsed_list = []
#         for entry in action_entries:
#             try:
#                 # 解析单个动作条目
#                 action_tuple = ast.literal_eval(entry.replace('"', "'"))
#
#                 # 标准化参数格式
#                 action_name, params = action_tuple[0]
#                 if not isinstance(params, tuple):
#                     params = (str(params),)
#
#                 # 重建标准格式
#                 parsed_list.append(((action_name, params), action_tuple[1]))
#             except:
#                 continue  # 跳过无法解析的条目
#
#         return parsed_list
#
#
# def ask_ds(client, prompt4sys, prompt4usr):
#     response = client.chat.completions.create(
#         model="deepseek-reasoner",
#         messages=[
#             {"role": "system", "content": prompt4sys},
#             {"role": "user", "content": prompt4usr},
#         ],
#         stream=False,
#         temperature=1.0
#     )
#     raw_output = response.choices[0].message.content  # deepseek-r1
#     try:
#         # 先尝试直接解析完整结构
#         return ast.literal_eval(raw_output)
#     except:
#         print("大模型返回格式出错!")
#
#
# def ask_llm(client, model_name, prompt4sys, prompt4usr):
#     print(prompt4sys)
#     print(prompt4usr)
#     completion = client.chat.completions.create(
#         # model="gpt-4.1",
#         model=model_name,
#         # messages=[
#         #     {
#         #         "role": "user",
#         #         "content": "Write a one-sentence bedtime story about a unicorn."
#         #     }
#         # ]
#         messages=[
#             {"role": "system", "content": prompt4sys},
#             {"role": "user", "content": prompt4usr},
#         ]
#     )
#     raw_output = completion.choices[0].message.content
#     print('大模型真正返回的结果:', raw_output)
#     try:
#         # 先尝试直接解析完整结构
#         return ast.literal_eval(raw_output)
#     except:
#         print("大模型返回格式出错!")
