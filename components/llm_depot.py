import math
import random
import yaml
# import dashscope
import ast
import re

from components.model_parser.parser_new import *
import itertools

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


def has_duplicates(input_list):
    # 使用set来存储已经见过的元素
    seen = set()
    # 使用set来存储重复的元素
    duplicates = set()

    for element in input_list:
        if element in seen:
            duplicates.add(element)
        else:
            seen.add(element)

    # 如果重复元素的集合非空，则表示有重复
    return bool(duplicates)


def applicable_actions(state, domain):
    """
    根据 domain（动作定义）、state（当前状态谓词）自动推理出所有可立即执行的动作（action_name, 参数元组）。
    - 内置类型继承关系（如 surface = [pallet, crate]，place = [depot, distributor]）。
    - 参数组合中不允许重复对象。
    """
    type_mapping = {
        'surface': ['pallet', 'crate'],
        'place': ['depot', 'distributor'],
        'locatable': ['truck', 'hoist', 'surface'],
        'object': ['place', 'locatable']
    }

    type_to_objects = {}
    for pred, args in state:
        for obj in args:
            obj_type = re.sub(r'\d+$', '', obj).lower()
            type_to_objects.setdefault(obj_type, set()).add(obj)

    def get_objects_for_type(tname):
        """返回类型 tname 及其所有子类型下的所有对象列表"""
        result = set()
        # 先看直接定义的对象
        if tname in type_to_objects:
            result.update(type_to_objects[tname])
        # 再看类型映射
        if tname in type_mapping:
            for subtype in type_mapping[tname]:
                result.update(get_objects_for_type(subtype))
        return list(result)

    applicable_actions = []

    # 遍历所有动作
    for action_name, action_def in domain.items():
        # 解析参数名、参数类型
        params = action_def.get("params", []) or action_def.get("params", [])
        param_names = [p[0].lstrip("?") for p in params]
        param_types = [p[1].lower() for p in params]
        # 生成每个参数的候选对象列表
        object_lists = []
        valid = True
        for t in param_types:
            candidates = get_objects_for_type(t)
            if not candidates:
                valid = False
                break
            object_lists.append(candidates)
        if not valid:
            continue
        # 枚举所有不重复对象的参数组合
        for combo in itertools.product(*object_lists):
            if len(set(combo)) < len(combo):
                continue  # 跳过有重复对象的
            # 检查前提条件
            all_prec_ok = True
            for prec in action_def.get("pos_prec", []):
                pred = prec[0]
                args = []
                for arg in prec[1]:
                    if isinstance(arg, str) and arg.startswith("?"):
                        pname = arg.lstrip("?")
                        idx = param_names.index(pname)
                        args.append(combo[idx])
                    else:
                        args.append(arg)
                if (pred, tuple(args)) not in state:
                    all_prec_ok = False
                    break
            if all_prec_ok:
                applicable_actions.append(Action(action_name, combo))
    return applicable_actions


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

    return set(new_state)  # {('truck', ('t1',)), ('at', ('p3', 'l1-1')), ('at', ('a1', 'l1-1')), ('location', ('l0-0',)), ('location', ('l1-0',)), ('in-city', ('l1-1', 'c1')), ('city', ('c0',)), ('obj', ('p4',)), ('at', ('t0', 'l0-0')), ('location', ('l1-1',)), ('location', ('l0-1',)), ('obj', ('p1',)), ('airplane', ('a0',)), ('airplane', ('a1',)), ('city', ('c1',)), ('at', ('a0', 'l0-1')), ('in-city', ('l1-0', 'c1')), ('at', ('p4', 'l1-0')), ('in-city', ('l0-1', 'c0')), ('obj', ('p3',)), ('airport', ('l1-1',)), ('airport', ('l0-1',)), ('truck', ('t0',)), ('at', ('p2', 'l0-1')), ('at', ('t1', 'l1-0')), ('obj', ('p2',)), ('in-city', ('l0-0', 'c0')), ('in', ('p1', 't0'))}


def goal_achieved(current_state, goal_state):
    """检查目标是否达成。"""
    for goal in goal_state:
        # 将目标状态格式化为元组
        goal_tuple = (goal[0], tuple(goal[1]))
        # 检查当前状态中是否存在该目标状态
        if goal_tuple not in current_state:
            return False
    return True


def back_propagate(node, reward, depth):
    """将奖励反向传播到路径上的所有节点。"""
    while node is not None:
        # 让mcts尽量兼顾到短深度的节点
        # node.visits += 1 + depth/(depth + 1)
        node.visits += 1
        node.reward += reward * (depth*100 + 2)/(depth*100 + 1)
        node = node.parent


def create_prompt(domain_description, init_state, current_state, goal_state, applied_actions, actions):
    prompt = f"Here is the data I have provided for you to analyse\n"
    prompt += f"{domain_description}\n"
    prompt += f"Init State: {init_state}\n"
    prompt += f"Current State: {current_state}\n"
    prompt += f"Goal State: {goal_state}\n"
    prompt += f"Action Trajectory: {', '.join([str((action.action_name, action.parameters)) for action in applied_actions])}\n"
    prompt += f"Available Actions & Parameters: {', '.join([str((action.action_name, action.parameters)) for action in actions])}\n"

    return prompt


def safe_parse_llm_output(output_text):
    # 阶段1：基础清洗
    cleaned = re.sub(r"\s+", " ", output_text)  # 合并多余空白
    cleaned = cleaned.replace("，", ",").replace("‘", "'").replace("’", "'")  # 统一符号

    # 阶段2：修复常见语法错误
    # 修复单元素元组缺少逗号的问题 (e.g. ('a') → ('a',))
    cleaned = re.sub(
        r"\(((?:'[^']+'(?:,\s*)?)+)\)",
        lambda m: f"({m.group(1).rstrip(',')},)" if len(m.group(1).split(',')) == 1 else f"({m.group(1)})",
        cleaned
    )

    # 新增阶段2.5：修复三元组结构错误
    # 将 (action, params, reward) 转换为 ((action, params), reward)
    cleaned = re.sub(
        r"\(\s*'([^']+)'\s*,\s*(\([^)]+\)|'[^']+')\s*,\s*(\d+)\s*\)",  # 匹配错误的三元组
        r"((\1, \2), \3)",  # 转换为正确的嵌套结构
        cleaned
    )

    # 阶段3：分层次解析
    try:
        # 先尝试直接解析完整结构
        return ast.literal_eval(cleaned)
    except:
        # 阶段4：容错解析
        action_entries = re.findall(
            r"(\(\(.*?\)\s*,\s*\d+\))",  # 匹配每个动作条目
            cleaned.replace("'", '"')  # 统一引号便于解析
        )

        parsed_list = []
        for entry in action_entries:
            try:
                # 解析单个动作条目
                action_tuple = ast.literal_eval(entry.replace('"', "'"))

                # 标准化参数格式
                action_name, params = action_tuple[0]
                if not isinstance(params, tuple):
                    params = (str(params),)

                # 重建标准格式
                parsed_list.append(((action_name, params), action_tuple[1]))
            except:
                continue  # 跳过无法解析的条目

        return parsed_list


def ask_ds(client, prompt4sys, prompt4usr):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt4sys},
            {"role": "user", "content": prompt4usr},
        ],
        stream=False,
        temperature=1.0
    )
    raw_output = response.choices[0].message.content  # deepseek-r1
    try:
        # 先尝试直接解析完整结构
        return ast.literal_eval(raw_output)
    except:
        print("大模型返回格式出错!")




# def ask_qwen(prompt):
#     dashscope.api_key = ai_key
#     response = dashscope.Generation.call(
#         model="qwen-max",
#         prompt=prompt
#     )
#     raw_output = response['output']['text']
#     print(f"raw_output: {raw_output}")
#
#     # 使用安全解析器
#     parsed_actions = safe_parse_llm_output(raw_output)
#
#     # 最终格式验证
#     validated = []
#     for action in parsed_actions:
#         try:
#             # 强制参数为元组
#             name, params = action[0]
#             if not isinstance(params, tuple):
#                 params = (params,)
#             validated.append(((name, params), action[1]))
#         except:
#             continue
#
#     return validated


# def mcts(model, iterations=500):
#     """Monte Carlo Tree Search 主循环"""
#     domain = model[DOMAIN]
#     init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#     goal = model[INSTANCE][GOAL]
#     with open('domain_des.yaml', 'r') as file:
#         domain_data = yaml.safe_load(file)
#     domain_description = domain_data.get('logistics', '')
#
#     root = Node(init_state)
#
#     times = 0
#     for _ in range(iterations):  # 进行iterations次节点拓展
#         # 选择
#         node = root
#         applied_actions = []
#         while node.children and node.is_fully_expanded(applicable_actions(node.state, domain)):
#             node = node.best_child()  # 定位到最佳路径上还未完全拓展完的节点
#             applied_actions.append(node.action)
#
#         if goal_achieved(node.state, goal):
#             print("goal achieved!!!")
#             break
#
#         prompt = create_prompt(domain_description, init_state, node.state, goal, applied_actions, applicable_actions(node.state, domain))
#         llm_list = ask_qwen(str(prompt))
#
#         # print(f"prompt: {prompt}")
#         times += 1
#         print(f"applicable actions: {[(action.action_name, action.parameters) for action in applicable_actions(node.state, domain)]}")
#         print(f"times: {times}, QWEN,action_list: {llm_list}")
#         print('-'*20)
#
#         # 扩展所有可执行动作 并且 反向传播
#         valid = bool(llm_list)
#         for assessed_action in llm_list:
#             action = Action(assessed_action[0][0], assessed_action[0][1])
#             reward = assessed_action[1] if valid else (2/5)
#             new_state = apply_action(node.state, action, domain)
#
#             child_node = Node(new_state, parent=node, action=action)
#             node.children.append(child_node)
#             if goal_achieved(new_state, goal):
#                 # 输出计划
#                 reward = 999
#             # 反向传播
#             back_propagate(child_node, reward, len(applied_actions))
#
#
#     # 从根节点选择最优路径
#     plan = []
#     node = root
#     while node.children:
#         node = node.best_child()
#         if node.action:
#             plan.append(str(node.action))
#
#     # 打印最终状态
#     final_state = root.state
#     for action in plan:
#         action_parts = action.split()
#         action_name = action_parts[0]
#         action_params = action_parts[1:]
#         final_state = apply_action(final_state, Action(action_name, action_params), domain)
#     print(f"Final State After Plan Execution: {final_state}")   #最终测试时开启
#
#     return plan



# if __name__ == "__main__":
#
#     domain_file = "depot/domain.pddl"
#     problem_file = "depot/instance-1.pddl"
#
#     model = parse_model(domain_file, problem_file)
#     domain = model[DOMAIN]
#     init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#     goal = model[INSTANCE][GOAL]
#
#     print("model:", model)
#     print("Parsed Domain:", domain)
#     print("Initial State:", init_state)
#     print("Init Predicates:", model[INSTANCE][INIT][PREDICATES])
#     print("Goal:", goal)
#
#     for applicable_action in applicable_actions(init_state, domain):
#         print(applicable_action)
#
#     # plan = components(model, 100)
#     # print("Generated Plan:")
#     # for step in plan:
#     #     print(step)
#     #
#     # with open("result_lt.txt", "w") as file:
#     #     for step in plan:
#     #         file.write(f"({step})\n")
