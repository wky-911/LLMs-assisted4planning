import math
import random
import yaml
# import dashscope
import ast
import re

from components.model_parser.parser_new import *
import itertools

exploration_weight = 0.5

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


def find_bindings(params, conditions, state):
    # params包含该动作所涉及的参数，conditions包含执行该动作所需的条件，state表示当前的状态
    # 例如DRIVE-TRUCK，该方法的参数有四个。由此params = [('?truck', 'object'), ('?loc-from', 'object'), ('?loc-to', 'object'), ('?city', 'object')]
    # 对于执行find_bindings后的bindings，其中的一组数据应该与('t1', 'l1-1', 'l1-2', 'c1')类似
    bindings = set()  # 装入所有可能的参数组合

    # 一、先绑定所有可组合的参数
    #     1.1抽取出需要的参数类型
    params = [p[0].replace('?','') for p in params]  # DRIVE-TRUCK ['?truck', '?loc-from', '?loc-to', '?city']
    #     1.2先将符合类型的参数存入参数列表
    #     1.2.1创建参数类型列表
    params_dict = [(param, set()) for param in params]
    #     1.2.2提取出条件中非介词的类型
    not_pos_names = [cond[0] for cond in conditions if cond[0] not in ('at', 'in', 'in-city')]  # ['truck', 'location', 'location', 'city']
    params_len = len(not_pos_names)  # 4
    #     1.2.3将当前状态中符合参数类型的对象存入参数列表
    for fact in state:
        for i in range(params_len):
            if fact[0] == not_pos_names[i]:
                params_dict[i][1].add(fact[1])  # [('truck', {('t1',), ('t0',), ('t2',)}), ('loc-from', {('l0-1',), ('l1-2',), ('l0-0',), ('l2-1',), ('l0-2',), ('l2-2',), ('l1-1',), ('l1-0',), ('l2-0',)}), ('loc-to', {('l0-1',), ('l1-2',), ('l0-0',), ('l2-1',), ('l0-2',), ('l2-2',), ('l1-1',), ('l1-0',), ('l2-0',)}), ('city', {('c0',), ('c1',), ('c2',)})]
    # print(f"param_dict: {params_dict}")
    #     1.3按照一定逻辑（loc_from != loc_to），将所有参数组合放入一个集合当中
    #     1.3.1检查not_pos_names中是否有重复元素.如果有,则元素不能相同
    if has_duplicates(not_pos_names):
        #     1.3.3将所有参数的组合放入一个集合中
        if params_len > 3:  # DRIVE-TRUCK
            params_combination = [
                (o1, o2, o3, o4)
                for o1 in params_dict[0][1]
                for o2, o3 in itertools.permutations(params_dict[1][1], 2)
                for o4 in params_dict[3][1]
            ]
        else:
            params_combination = [
                (o1, o2, o3)
                for o1 in params_dict[0][1]
                for o2, o3 in itertools.permutations(params_dict[1][1], 2)
            ]
        # print(f"params_combination: {params_combination}, params_combination_len: {len(params_combination)}")
    #     1.3.2如果没有重复约束，直接生成所有可能组合
    else:
        params_combination = [
            (o1, o2, o3)
            for o1 in params_dict[0][1]
            for o2 in params_dict[1][1]
            for o3 in params_dict[2][1]
        ]

    # 二、检查参数组合是否满足介词。如果有就保存，否则跳过。
    #     1.1提取当前状态下的所有介词
    cond_names = [cond[0] for cond in conditions if cond[0] in ('at', 'in', 'in-city')]  # 提取介词谓语
    facts = set([fact for fact in state if fact[0] in cond_names]) # 从当前状态下可行的分支出发
    # print(f"params: {params}, cond: {cond_names}, facts: {facts}")
    #     1.2检验所有组合,判断一个组合是否能够符合所有的介词要求
    for combination in params_combination:
        # 1.2.1构造字典 {param:value}
        temp_dict = []  # [('truck', ('t1',)), ('loc-from', ('l0-2',)), ('loc-to', ('l2-2',)), ('city', ('c1',))]
        for index in range(params_len):
            temp_dict.append((params[index], combination[index]))
        # 1.2.2 检验是否满足初始状态的要求--所有介词要求在初始状态里同时被满足
        # 基于该组合,构造新的动作条件; 该条件的对象是具体的.
        replica_conditions = [tuple_to_list(cond) for cond in conditions if cond[0] in ('at', 'in', 'in-city')]
        for cond in replica_conditions:
            # 1.2.3如果条件的第二个名字去掉? 与temp_dict的第一个元素相同,则填入新条件
            for temp in temp_dict:
                if cond[1][0].replace('?', '') == temp[0]:
                    cond[1][0] = temp[1][0]
                if cond[1][1].replace('?', '') == temp[0]:
                    cond[1][1] = temp[1][0]
        replica_conditions = list(list_to_tuple(r_cond) for r_cond in replica_conditions)  # [('at', ('t0', 'l2-1')), ('in-city', ('l2-1', 'c0')), ('in-city', ('l0-1', 'c0'))]
        # print(f"replica_conditions: {replica_conditions}")
        #     1.3如果该组合的所有条件都能在当前状态中被满足,则将其保存下来
        valid = True
        for cond in replica_conditions:
            if cond not in facts:
                valid = False
        if valid:
            bindings.add(combination)
    # print(f"bindings: {bindings}")
    return bindings


def applicable_actions(state, domain):
    """获取在当前状态下可以执行的动作列表。"""
    actions = []
    for action_name, action_info in domain.items():
        conditions = [list_to_tuple(cond) for cond in action_info[POS_PREC]]  # 取出某个动作的执行条件
        parameters = [param for param in action_info[PARARMETERS]]  # 取出该动作的参数
        bindings = find_bindings(parameters, conditions, state)  # 尝试为该动作找到可行的具体参数（该状态下是否可执行该动作）
        if bindings is not None:  # 如果找到有效绑定，则该动作可执行
            for binding in bindings:
                actions.append(Action(action_name, tuple(param[0] for param in binding)))
    random.shuffle(actions)
    return list(actions)  # {('FLY-AIRPLANE', ('a2', 'l2-0', 'l1-0')), ('LOAD-AIRPLANE', ('p1', 'a0', 'l0-0')), ('FLY-AIRPLANE', ('a0', 'l0-0', 'l2-0')), ('FLY-AIRPLANE', ('a0', 'l0-0', 'l1-0')), ('LOAD-TRUCK', ('p3', 't1', 'l1-2')), ('DRIVE-TRUCK', ('t1', 'l1-2', 'l1-1', 'c1')), ('LOAD-AIRPLANE', ('p7', 'a2', 'l2-0')), ('DRIVE-TRUCK', ('t0', 'l0-2', 'l0-0', 'c0')), ('FLY-AIRPLANE', ('a1', 'l1-0', 'l0-0')), ('FLY-AIRPLANE', ('a1', 'l1-0', 'l2-0')), ('FLY-AIRPLANE', ('a2', 'l2-0', 'l0-0')), ('DRIVE-TRUCK', ('t0', 'l0-2', 'l0-1', 'c0')), ('LOAD-TRUCK', ('p6', 't2', 'l2-2')), ('DRIVE-TRUCK', ('t2', 'l2-2', 'l2-0', 'c2')), ('LOAD-TRUCK', ('p0', 't0', 'l0-2')), ('LOAD-AIRPLANE', ('p4', 'a1', 'l1-0')), ('DRIVE-TRUCK', ('t1', 'l1-2', 'l1-0', 'c1')), ('DRIVE-TRUCK', ('t2', 'l2-2', 'l2-1', 'c2'))}


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
    action_bindings ={f"{param}": value for param, value in zip(params, action.parameters)}  # {'airplane': 'a1', 'loc-from': 'l1-1', 'loc-to': 'l0-1'}
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
    # print(f"raw_output: {raw_output}")

    # 使用安全解析器
    parsed_actions = safe_parse_llm_output(raw_output)

    # 最终格式验证
    validated = []
    for action in parsed_actions:
        try:
            # 强制参数为元组
            name, params = action[0]
            if not isinstance(params, tuple):
                params = (params,)
            validated.append(((name, params), action[1]))
        except:
            continue

    return validated




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
#
#
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
#
#
#
# if __name__ == "__main__":
#     domain_file = "logistics/domain.pddl"
#     problem_file = "logistics/problem_simple.pddl"
#
#     model = parse_model(domain_file, problem_file)
#
#     plan = mcts(model, 100)
#     print("Generated Plan:")
#     for step in plan:
#         print(step)
#
#     with open("result_lt.txt", "w") as file:
#         for step in plan:
#             file.write(f"({step})\n")
