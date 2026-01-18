import json
import copy
import time
import paramiko
import pandas as pd
from sympy.physics.units import action
from openai import OpenAI

from val.validate import validate_plan
from collections import defaultdict, deque
from components.llm_blocksworld import *
# The 'directory' corresponds to the directory where the instance of the Blocks domain resides in the Linux system.
directory='blocksworld'
# This corresponds to the domain name defined in the "domain.pddl".
domain='BLOCKS'

"""
1.Call the DW itself to plan the total instances, with a time limit of 10 seconds.  √
------------------------------------------
2.Break down the target state, and the planning time for each sub-instance is limited to 30 seconds.  √
-------------------------------------------
3.For sub-instances, if the time exceeds 30 seconds, execute 5 steps of the LLM&MCTS algorithm.  √
4.After obtaining the new state, call DW and set the planning time limit to 30 seconds.  √
-------------------------------------------
5.If the operation fails, the LLM will be invoked to further split the sub-instance, with the planning time limited to 30 seconds.  √
"""

# ds_key = os.getenv('DS_API_KEY')
# client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
# model_name = "claude-sonnet-4-20250514"
model_name = "gpt-5"
client = OpenAI(
    api_key=os.environ["METACHAT_API_KEY"],
    base_url="https://llm-api.mmchat.xyz/v1"
)

class InsList:
    def __init__(self):
        self.ins_children = []

    # 在最后位置插入子实例
    def add_child(self, child):
        if self.ins_children:
            last_child = self.ins_children[-1]
            last_num = last_child.get_num()
            child.set_num(last_num + 1)
        else:
            child.set_num(0)
        self.ins_children.append(child)

    def get_all_time(self):
        return sum(child.get_time() for child in self.ins_children)

    def get_children(self, num):
        return self.ins_children[num]

    def __str__(self):
        s = ""
        for child in self.ins_children:
            s += f"num: {child.get_num()}, goal: {child.get_goal()}\n"
        return s


class InsChild:
    def __init__(self, goal, cur=None, num=0, actions=None, time=None):
        self.goal = goal
        self.cur = cur
        self.num = num
        self.actions = actions
        self.time = time

    def set_cur(self, cur):
        self.cur = cur

    def set_num(self, num):
        self.num = num

    def set_actions(self, actions_str):
        if actions_str:
            lines = actions_str.strip().split('\n')
            actions = []
            for line in lines:
                match = re.match(r'\(\s*([^\s()]+)\s*([^)]*)\)', line)
                if match:
                    action_name = match.group(1)  # 动作名，可以包含中划线
                    params_str = match.group(2).strip()
                    parameters = params_str.split() if params_str else []
                    action = Action(action_name, tuple(parameters))
                    actions.append(action)
            self.actions = actions

    def set_time(self, time):
        self.time = time

    def get_cur(self):
        return self.cur

    def get_goal(self):
        return self.goal

    def get_num(self):
        return self.num

    def get_actions(self):
        return self.actions

    def get_time(self):
        return self.time

    def pull_actions(self):
        for action in self.actions:
            print(action)

    def __str__(self):
        return f"num = {self.num}, goal = {self.goal}"

    def get_instance(self, domain_name="BLOCKS", instance_name="temp_instance"):
        """
        Generate a PDDL problem string from the current state (cur) and goal state (goal)
        :param domain_name: Domain name
        :param instance_name: Problem name
        :return: PDDL problem string
        """
        # 1. 自动提取所有涉及的对象，只考虑参数非空的谓词
        objects = sorted({arg for pred in self.cur if isinstance(pred, (tuple, list)) and len(pred) == 2
                          for arg in (pred[1] if isinstance(pred[1], (tuple, list)) else [pred[1]])
                          if isinstance(arg, str)})

        # 2. 通用谓词转字符串
        def pred_to_str(pred):
            if isinstance(pred, (tuple, list)) and len(pred) == 2:
                name, args = pred
                if isinstance(args, (tuple, list)):
                    if len(args) > 0:
                        return f"({name} {' '.join(args)})"
                    else:
                        return f"({name})"
                elif isinstance(args, str):
                    return f"({name} {args})"
                else:
                    return f"({name})"
            elif isinstance(pred, str):
                return f"({pred})"
            else:
                return str(pred)  # fallback

        # 3. 生成PDDL初始状态与目标
        init_preds = [pred_to_str(pred) for pred in self.cur]
        # 目标状态兼容单一和多个谓词（建议上游统一格式，这里做适配）
        if isinstance(self.goal[0], str):
            goal_preds = [pred_to_str(self.goal)]
        else:
            goal_preds = [pred_to_str(pred) for pred in self.goal]

        # 4. 拼接字符串
        s = ""
        s += f"(define (problem {instance_name})\n"
        s += f"  (:domain {domain_name})\n"
        s += "  (:objects " + " ".join(objects) + ")\n"
        s += "  (:init\n"
        for item in init_preds:
            s += f"    {item}\n"
        s += "  )\n"
        s += "  (:goal\n"
        if len(goal_preds) == 1:
            s += f"    {goal_preds[0]}\n"
        else:
            s += "    (and\n"
            for item in goal_preds:
                s += f"      {item}\n"
            s += "    )\n"
        s += "  )\n"
        s += ")\n"
        return s


# 针对于blocksworld领域，得到一个按执行顺序排列的子目标状态
def sort_blocksworld_goals(goal_list):
    edges = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()
    for goal in goal_list:
        _, [x, y] = goal  # x在y上
        edges[y].append(x)
        in_degree[x] += 1
        nodes.add(x)
        nodes.add(y)
        if y not in in_degree:
            in_degree[y] = 0

    # 1. 找所有塔底部
    bottoms = [node for node in nodes if in_degree[node] == 0]

    # 2. 拓扑排序
    queue = deque(bottoms)
    result = []
    on_dict = {(x, y): goal for goal in goal_list for x, y in [goal[1]]}

    while queue:
        cur = queue.popleft()
        for nxt in edges[cur]:
            if (nxt, cur) in on_dict:
                result.append(on_dict[(nxt, cur)])
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # 3. 按照顺序在前面插入 ontable 目标
    result = [ ['ontable', [bottom]] for bottom in bottoms ] + result

    return result


# 得到子实例序列
def get_ins_list(sorted_goal):
    """
    Given the instance goal state, obtain an ordered list of sub-goals to achieve
    :param goal_list:
    :return: ins_list
    """
    ins_list = InsList()
    for goal in sorted_goal:
        ins_list.add_child(InsChild(goal))
    return ins_list


# 将子实例写入ubuntu系统中的文件内，用于DW规划
def write_instance_to_remote(instance_content, directory=directory):
    """
    Write the instance to an Ubuntu machine for DW planning
    :param instance_content:
    :return:
    """
    hostname = "192.168.19.130"
    username = "xiyou"
    password = "xiyou..9911"
    remote_dir = f"/home/xiyou/FastDownward/{directory}"
    filename = "instance_temp.pddl"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        # 连接SSH
        ssh.connect(hostname, username=username, password=password)
        # 打开SFTP通道
        sftp = ssh.open_sftp()
        # 拼接完整路径
        remote_path = f"{remote_dir}/{filename}"
        # 写入文件
        with sftp.file(remote_path, 'w') as f:
            f.write(instance_content)
        sftp.close()
    except Exception as e:
        print("写入远程文件失败:", e)
    finally:
        ssh.close()


# 调用DW规划器进行规划
def planner(timeout, domain_name):
    hostname = "192.168.19.130"
    username = "xiyou"
    password = "xiyou..9911"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)

    command = f"cd /home/xiyou/FastDownward && python3 sub-instance-planner.py {timeout} {domain_name}"

    stdin, stdout, stderr = ssh.exec_command(command)
    output_text = stdout.read().decode()
    error_text = stderr.read().decode()
    # print(output_text)

    if error_text:
        print("规划失败!!", error_text)
        ssh.close()
        return 'fail', None
    else:
        plan_marker = "读取 sas_plan 文件内容："
        planner_time_marker = "INFO     Planner time: "

        # 处理规划结果
        if plan_marker in output_text:
            plan_part = output_text.split(plan_marker, 1)[1].strip()
            plan_lines = plan_part.splitlines()
            cleaned_plan = "\n".join(plan_lines[:-1])
        else:
            print("读取 sas_plan 文件异常！")
            ssh.close()
            return 'fail', None
        # 处理规划时间
        if planner_time_marker in output_text:
            planner_time_part = output_text.split(planner_time_marker, 1)[1].strip()
            planner_time = planner_time_part.split('\n', 1)[0]
            print(f"规划时间: {planner_time}")
        else:
            print("未找到规划时间信息")
            planner_time = None

    ssh.close()
    return cleaned_plan, planner_time


#调用DW解决实例，限定时间为10s
def solve_goal_dw(init_state, goal_state, write_instance_to_remote, planner, timeout=10, directory=directory, domain_name=domain):
    instance = InsChild(goal_state)
    instance.set_cur(init_state)
    instance_temp = instance.get_instance(domain_name)
    write_instance_to_remote(instance_temp)
    plan, planner_time = planner(timeout, directory)
    if plan == 'fail':
        print("plan fail!")
        return plan, None
    print('plan success!')
    instance.set_actions(plan)
    instance.set_time(planner_time)
    actions = instance.get_actions()
    return actions, planner_time


# 调用DW解决所有的子实例
def solve_sub_goals_dw(ins_list, init_state, domain, write_instance_to_remote, planner, timeout=30, directory=directory, domain_name=domain):
    """
    Solve each sub-goal in order, automatically propagate the state, and merge all actions into the final plan
    :param ins_list: InsList object containing all sub-goals in order
    :param init_state: Initial state (list/tuple/set)
    :param domain: Domain description (the domain part from parse_model)
    :param write_instance_to_remote: Function to write the instance file
    :param planner: Function to invoke the remote planner
    :return: total_actions (final full action sequence: list[Action])
    """
    cur_state = init_state
    total_actions = []
    for i in range(len(ins_list.ins_children)):
        child = ins_list.get_children(i)
        child.set_cur(cur_state)  # 将最新状态设置为当前子目标的初始状态
        temp_instance = child.get_instance(domain_name)
        write_instance_to_remote(temp_instance)
        plan, planner_time = planner(timeout, directory)
        print(f"单目标规划成功，时长为: {planner_time}") if planner_time else None
        if plan == 'fail':
            print("plan fail!")
            return plan, None
        child.set_actions(plan)
        child.set_time(float(planner_time.strip('s')))
        actions = child.get_actions()
        if actions:
            total_actions.extend(actions)
            # 用该子目标所有action更新当前状态，为下一个子目标做准备
            for action in actions:
                cur_state = apply_action(cur_state, action, domain)
        else:
            print(f"子目标{i}已经实现。。")
    return total_actions, ins_list.get_all_time()



def create_prompt4action(current_state, goal_state, applied_actions, actions):
    prompt4usr = f"current state: {current_state}\n"
    prompt4usr += f"goal state: {goal_state}\n"
    prompt4usr += f"action trajectory: {', '.join([str((action.action_name, action.parameters)) for action in applied_actions])}\n"
    prompt4usr += f"applicable actions: {', '.join([str((action.action_name, action.parameters)) for action in actions])}\n"
    prompt4usr += f"Please conduct a thorough analysis based on the provided current state, goal state, action trajectory, and applicable actions, and then recommend the most promising action."
    return prompt4usr


# 创建用于大模型预测规划中间态的提示词模板
def create_prompt4decompose(current_state, goal_state):
    prompt = f"当前状态: {current_state}\n"
    prompt += f"目标状态: {goal_state}\n"
    prompt += f"请返回介于当前状态和目标状态之间的一个中间状态。"

    return prompt


def ask4action(current_state, goal_state, applied_actions, actions, domain_name=directory):
    # with open('configs/domain_des3.yaml', 'r', encoding='utf-8') as file:
    with open('configs/domain_des5.yaml', 'r', encoding='utf-8') as file:
        domain_data = yaml.safe_load(file)
    prompt4sys = domain_data.get(domain_name, '')
    prompt4usr = create_prompt4action(current_state, goal_state, applied_actions, actions)

    # action_tuple = ask_ds(client, prompt4sys, prompt4usr)
    action_tuple = ask_llm(client, model_name, prompt4sys, prompt4usr)
    # print(action_tuple)
    # print(f"动作元组为: {action_tuple[0]}, {action_tuple[1]}")
    try:
        action = Action(action_tuple[0], action_tuple[1])
        return action
    except:
        print(f"动作包装失败! 大模型返回的动作为: {action_tuple}")
        return


# 访问大模型，预测当前状态到目标状态的中间态
def ask4decompose(prompt4usr, domain_name=directory):
    with open('configs/decomposition.yaml', 'r', encoding='utf-8') as file:
        domain_data = yaml.safe_load(file)
    prompt4sys = domain_data.get(domain_name, '')

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt4sys},
            {"role": "user", "content": prompt4usr},
        ],
        stream=False,
        temperature=1.0
    )
    raw_output = response.choices[0].message.content
    print(f"raw_output: {raw_output}")

    return raw_output


def solve_sub_goals_dw_v2(init_state, goal_state, domain, obj,
                          write_instance_to_remote, planner,
                          timeout=30, directory=directory, domain_name=domain,
                          instance_index=None):
    """
    Each time the LLM outputs only one action; after executing it, immediately try DW. Up to 10 jumps.
    After planning succeeds, pass the LLM call count into val_and_write.
    """
    # 先尝试全局直接规划
    actions, planner_time = solve_goal_dw(
        init_state, goal_state, write_instance_to_remote, planner,
        timeout=10, directory=directory, domain_name=domain_name
    )

    call_count = 0  # 记录LLM调用次数

    if actions != 'fail':
        # 全局规划成功
        total_actions = actions
        try:
            all_time = float(planner_time) if planner_time is not None else 0.0
        except ValueError:
            all_time = float(planner_time.replace('s', '').strip()) if planner_time else 0.0
        plan_status = 'success'
    else:
        # 全局规划失败，按子目标分解
        print("全局规划失败，拆解目标状态并逐一规划子目标...")
        sorted_goal = sort_blocksworld_goals(goal_state) if domain_name == 'BLOCKS' else goal_state
        ins_list = get_ins_list(sorted_goal) # 分解目标
        cur_state = set(init_state)
        total_actions = []
        all_time_val = 0.0
        plan_status = 'success'
        for i in range(len(ins_list.ins_children)):
            child = ins_list.get_children(i)
            child.set_cur(cur_state)
            temp_instance = child.get_instance(domain_name)
            write_instance_to_remote(temp_instance)
            plan_part, planner_time_part = planner(timeout, directory)
            if plan_part != 'fail':
                # DW直接成功
                child.set_actions(plan_part)
                child.set_time(planner_time_part)
                actions_part = child.get_actions()
                if actions_part:
                    total_actions.extend(actions_part)
                    for act in actions_part:
                        cur_state = apply_action(cur_state, act, domain)
                if planner_time_part:
                    try:
                        time_val = float(planner_time_part)
                    except ValueError:
                        time_val = float(planner_time_part.replace('s', '').strip()) if isinstance(planner_time_part, str) else 0.0
                    all_time_val += time_val
            else:
                # DW失败：最多跳跃10次，每次LLM只给1个动作，步步重试DW
                print(f"子目标规划失败! 跳跃尝试中... 子目标: {child.get_goal()}")
                jump_success = False
                applied_actions = [] # 初始化动作轨迹
                for attempt in range(10):
                    call_count += 1
                    # jump_plan = ask4plan(domain, obj, cur_state, child.get_goal(), iterations=1)
                    applicable_action_list = applicable_actions(cur_state, domain, obj)
                    print(f"当前状态: {cur_state}")
                    print(f"目标状态: {child.get_goal()}")
                    print(f"动作轨迹: {[(action.action_name, action.parameters) for action in applied_actions]}")
                    print(f"applicable actions: {[(action.action_name, action.parameters) for action in applicable_action_list]}")
                    # TODO--------请求LLM---------
                    jump_action = ask4action(cur_state, child.get_goal(), applied_actions, applicable_action_list)
                    print(f"times: {call_count}, jump_action: {jump_action}")
                    if not jump_action :
                        # LLM完全没出动作
                        plan_status = 'fail'
                        break
                    # 通常jump_plan只有1个动作
                    cur_state = apply_action(cur_state, jump_action, domain)
                    total_actions.append(jump_action)
                    applied_actions.append(jump_action)
                # TODO---------------------------------------------------------------------
                    # 用新状态再试DW
                    child.set_cur(cur_state)
                    temp_instance2 = child.get_instance(domain_name)
                    write_instance_to_remote(temp_instance2)
                    plan_part2, planner_time_part2 = planner(timeout, directory)
                    if plan_part2 != 'fail':
                        # 跳跃+DW成功
                        jump_success = True
                        print(f"跳跃第{attempt+1}步后DW成功, 耗时: {planner_time_part2}")
                        child.set_actions(plan_part2)
                        child.set_time(planner_time_part2)
                        actions_part2 = child.get_actions()
                        if actions_part2:
                            total_actions.extend(actions_part2)
                            for act in actions_part2:
                                cur_state = apply_action(cur_state, act, domain)
                        if planner_time_part2:
                            try:
                                time_val2 = float(planner_time_part2)
                            except ValueError:
                                time_val2 = float(planner_time_part2.replace('s', '').strip()) if isinstance(planner_time_part2, str) else 0.0
                            all_time_val += time_val2
                        break  # 跳出跳跃循环，进入下一个子目标
                if not jump_success:
                    # 10次都失败
                    print("10步跳跃后DW仍失败，任务终止。")
                    plan_status = 'fail'
                    break  # 终止所有子目标循环
        all_time = all_time_val

    # 结果输出
    if plan_status == 'fail':
        if instance_index is not None:
            with open("plan_detail.txt", "a") as file:
                file.write(f"规划失败！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        return 'fail', all_time
    else:
        val_and_write('plan_results.csv', total_actions, instance_index, all_time, call_count)
        return total_actions, all_time


def solve_sub_goals_dw_v3(init_state, goal_state, domain,
                          write_instance_to_remote, planner,
                          timeout=30, directory=directory, domain_name=domain,
                          instance_index=None):
    """
    Try to solve the full planning problem step by step (may contain multiple sub-goals), v3:
    1. First call DW to solve the overall goal within 10s; if it succeeds, return the plan directly.
    2. If it fails, split the overall goal into an ordered set of sub-goal instances (for decomposable domains such as Blocksworld) and plan them one by one (30s timeout each).
    3. If a sub-goal fails to plan, call ask4decompose to get an intermediate state between the current state and that sub-goal, and split the sub-goal into two sub-instances:
            - Sub-instance 1: start = current state, goal = intermediate state;
            - Sub-instance 2: start = the new state after finishing sub-instance 1, goal = the original goal.
          Replace the original failed instance with these two new instances (do not keep the original).
    4. Re-plan the new sub-instances (10s timeout each); if it still fails, repeat splitting up to 10 iterations.
    5. After all sub-instances are planned successfully, merge all actions into the final plan and validate it. On success, write to pfile.plan and log success to plan_detail.txt; otherwise log failure.
    :param init_state: Initial state (list/tuple/set; elements are predicate tuples)
    :param goal_state: Full goal state (list/tuple/set; elements are predicate tuples)
    :param domain: Domain model description (the domain part from parse_model)
    :param obj: Object collection/list for the problem (not explicitly used; kept for interface consistency)
    :param write_instance_to_remote: Function to write the instance to the remote planner
    :param planner: Function to invoke the remote planner
    :param timeout: Timeout per DW call during the sub-goal stage (seconds), default 30
    :param directory: Directory name for the planning task on the remote planner, default 'blocksworld'
    :param domain_name: PDDL domain name used to generate the instance file, default 'BLOCKS'
    :param instance_index: Instance index (used for logging and validation), optional
    :return: (total_actions, all_time), where total_actions is the final action sequence (list[Action] or 'fail') and all_time is the total planning time (seconds)
    """
    # 尝试直接用DW规划完整目标（限制时间10秒）
    actions, planner_time = solve_goal_dw(init_state, goal_state,
                                         write_instance_to_remote, planner,
                                         timeout=10, directory=directory, domain_name=domain_name)
    if actions != 'fail':
        # 全局规划成功，返回完整计划
        total_actions = actions
        # 转换规划耗时为浮点数
        try:
            all_time = float(planner_time) if planner_time is not None else 0.0
        except ValueError:
            all_time = float(planner_time.replace('s', '').strip()) if planner_time else 0.0

        val_and_write('plan_results.csv', total_actions, instance_index, all_time)
        # # 将计划写入文件并验证
        # plan_file = "pfile.plan"
        # open(plan_file, 'w').close()  # 清空原文件内容
        # for act in total_actions:
        #     with open(plan_file, "a") as file:
        #         file.write(f"{act}\n")
        # # 验证计划（如果提供了实例编号）
        # if instance_index is not None:
        #     import os
        #     project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        #     domain_file = os.path.join(project_dir, f"ipc_instances/domain_{directory}.pddl")
        #     problem_file = os.path.join(project_dir, f"ipc_instances/{directory}/instance-{instance_index}.pddl")
        #     result = validate_plan(domain_path=domain_file, problem_path=problem_file, plan_path=plan_file)
        #     # 根据验证结果记录日志
        #     with open("plan_detail.txt", "a") as file:
        #         if not result["is_valid"]:
        #             file.write(f"规划失败！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        #             return 'fail', all_time
        #         else:
        #             file.write(f"**规划成功！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        return total_actions, all_time

    # 全局规划失败，进行子目标拆解规划
    print("全局规划失败，拆解目标状态并逐一规划子目标...")
    sorted_goal = sort_blocksworld_goals(goal_state) if domain_name == 'BLOCKS' else goal_state
    ins_list = get_ins_list(sorted_goal)  # 构建子目标实例列表
    cur_state = set(init_state)          # 当前状态（用集合便于状态更新）
    total_actions = []
    all_time_val = 0.0
    plan_status = 'success'
    sub_timeout = timeout  # 子目标规划超时时间（初始为传入的timeout，一般30秒）

    # 最多允许拆分10次
    max_splits = 10
    split_count = 0
    # 起始规划的子目标序号
    start_index = 0

    # 循环尝试逐个规划子目标，必要时拆分
    while split_count < max_splits:
        plan_status = 'success'
        # 从当前未完成的子目标开始继续规划
        for idx in range(start_index, len(ins_list.ins_children)):
            child = ins_list.get_children(idx)
            child.set_cur(cur_state)
            temp_instance = child.get_instance(domain_name)
            write_instance_to_remote(temp_instance)
            plan_part, planner_time_part = planner(sub_timeout, directory)
            if plan_part != 'fail':
                # 子目标规划成功，记录动作序列和时间
                child.set_actions(plan_part)
                child.set_time(planner_time_part)
                actions_part = child.get_actions()
                # 累积动作序列并更新当前状态
                if actions_part:
                    total_actions.extend(actions_part)
                    for act in actions_part:
                        cur_state = apply_action(cur_state, act, domain)
                # 累积规划时间
                if planner_time_part:
                    try:
                        time_val = float(planner_time_part)
                    except ValueError:
                        time_val = float(planner_time_part.replace('s', '').strip()) if isinstance(planner_time_part, str) else 0.0
                    all_time_val += time_val
                # 继续下一个子目标
                continue
            else:
                # 子目标规划失败，需大模型辅助拆分
                plan_status = 'fail'
                failing_index = idx
                print(f"当前状态为 {child.get_cur()}")
                print(f"子目标规划失败，尝试拆分子目标 {child.get_goal()} ...")
                # 准备提示词并调用大模型获取中间状态
                prompt = create_prompt4decompose(cur_state, child.get_goal())
                intermediate_str = ask4decompose(prompt)
                print("intermediate_str:", intermediate_str)
                try:
                    intermediate_preds = json.loads(intermediate_str)
                except Exception as e:
                    # 中间状态解析失败，终止规划
                    print("中间状态解析失败:", e)
                    # 记录失败状态
                    plan_status = 'fail'
                    break
                print(intermediate_preds, type(intermediate_preds))

                # 构造新的两个子目标实例
                child1_goal = intermediate_preds # 中间态为新目标
                child2_goal = copy.deepcopy(child.get_goal())
                child1 = InsChild(child1_goal)
                child2 = InsChild(child2_goal)
                # 在列表中替换失败实例为两个新实例
                ins_list.ins_children[failing_index] = child1
                ins_list.ins_children.insert(failing_index + 1, child2)
                # 重新编号子实例序号（保持与列表索引一致）
                for j, ch in enumerate(ins_list.ins_children):
                    ch.set_num(j)
                # 更新拆分计数和新的起始索引（从第一个新子目标开始规划）
                split_count += 1
                start_index = failing_index
                # 一旦发生拆分，后续规划超时时间调整为10秒
                sub_timeout = 10
                # 跳出当前循环，进入下一轮迭代重新规划新的子目标序列
                break
        # 内层循环正常结束（无失败）或因失败拆分跳出后，检查状态
        if plan_status == 'success':
            # 所有子目标均成功，无需继续拆分
            break
        if plan_status == 'fail' and split_count >= max_splits:
            # 已达到最大拆分次数，仍未成功
            break
        # 若当前循环因为拆分跳出，继续下一轮 while 尝试从新子目标开始规划
        continue

    # 规划结束，依据结果记录输出
    all_time = all_time_val
    if plan_status == 'fail':
        # 写入失败结果到 plan_detail.txt
        if instance_index is not None:
            with open("plan_detail.txt", "a") as file:
                file.write(f"规划失败！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        return 'fail', all_time
    else:
        val_and_write('plan_results.csv', total_actions, instance_index, all_time, split_count)
        # # 输出完整计划到文件并验证
        # plan_file = "pfile.plan"
        # open(plan_file, 'w').close()
        # for act in total_actions:
        #     with open(plan_file, "a") as file:
        #         file.write(f"{act}\n")
        # if instance_index is not None:
        #     import os
        #     project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        #     domain_file = os.path.join(project_dir, f"ipc_instances/domain_{directory}.pddl")
        #     problem_file = os.path.join(project_dir, f"ipc_instances/{directory}/instance-{instance_index}.pddl")
        #     result = validate_plan(domain_path=domain_file, problem_path=problem_file, plan_path=plan_file)
        #     with open("plan_detail.txt", "a") as file:
        #         if not result["is_valid"]:
        #             file.write(f"规划失败！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        #             return 'fail', all_time
        #         else:
        #             file.write(f"**规划成功！实例序号为: {instance_index}; The planning time: {all_time}s\n")
        return total_actions, all_time


def val_and_write(csv_file, final_actions, index, all_time, call_count=0):
    # 判断表格文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 定义要写入的一行内容（每次一个实例）
    if final_actions == 'fail':
        result_row = {
            "实例序号": index,
            "规划时间": all_time,
            "计划步骤数": "fail"
        }
    else:
        plan_file = 'pfile.plan'
        open(plan_file, 'w').close()
        for a in final_actions:
            with open(plan_file, "a") as file:
                file.write(f"{a}\n")

        # VAL审核
        result = validate_plan(
            domain_path=domain_file,
            problem_path=problem_file,
            plan_path=plan_file,
        )

        if result["is_valid"]:
            result_row = {
                "实例序号": index,
                "规划时间": all_time,
                "计划步骤数": len(final_actions),
                "调用大模型次数": call_count,
                "领域": directory
            }
        else:
            result_row = {
                "实例序号": index,
                "规划时间": "fail",
                "计划步骤数": "fail"
            }

    # 写入csv（如果没有表头先写表头，否则直接追加）
    df = pd.DataFrame([result_row])
    if not file_exists:
        df.to_csv(csv_file, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(csv_file, index=False, mode='a', header=False, encoding='utf-8-sig')


# # DW与llm预测结合
# if __name__ == '__main__':
#     # index_range = [33, 36] # 成功
#     # index_range = [38, 42, 43, 44, 47, 48, 49] # 42, 43 规划失败; 其余成功
#     # index_range = [50, 51, 52, 53, 54, 56, 57, 58, 59, 60] # 56，58，60规划失败；其余成功
#     # index_range = [61, 62] # 62规划失败
#     # index_range = [63, 64, 65, 66, 67] # 65成功，其余失败
#     # index_range = [68, 69, 70]
#     # index_range = [33, 36, 38, 42, 43, 44, 47, 48, 49, 50]
#     index_range = [33]
#     for index in index_range:
#         domain_name = 'blocksworld'
#         plan_detail = 'plan_detail.txt'
#         project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#         domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
#         problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")
#
#         model = parse_model(domain_file, problem_file)
#         domain = model[DOMAIN]
#         init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#         goal = model[INSTANCE][GOAL]
#         plan, planner_time = solve_sub_goals_dw_v3(
#             init_state,
#             goal,
#             domain,
#             write_instance_to_remote,
#             planner,
#             30,
#             domain_name,
#             'BLOCKS',
#             index
#         )
#         print([(action.action_name, action.parameters) for action in plan]) if plan != 'fail' else print(plan)

# # 大模型预测中间态
# if __name__ == '__main__':
#     init_state1 = {('ontable', ('d',)), ('ontable', ('k',)), ('clear', ('m',)), ('on', ('q', 'b')), ('clear', ('l',)), ('clear', ('k',)), ('ontable', ('c',)), ('on', ('g', 'q')), ('ontable', ('a',)), ('clear', ('c',)), ('on', ('h', 'j')), ('on', ('m', 'd')), ('clear', ('a',)), ('handempty', ()), ('on', ('b', 'p')), ('clear', ('n',)), ('on', ('j', 'o')), ('on', ('n', 'f')), ('ontable', ('f',)), ('on', ('i', 'e')), ('clear', ('i',)), ('on', ('e', 'g')), ('on', ('p', 'h')), ('ontable', ('o',)), ('ontable', ('l',))}
#     goal1 = [["ontable", ["h"]], ["clear", ["j"]]]
#     prompt = create_prompt4decompose(init_state1, goal1)
#     print(prompt)
#     print(ask4decompose(prompt))


# # DW与llm启发结合
# if __name__ == '__main__':
#     # index_range = [33, 36, 38, 42, 43, 44, 47, 48, 49, 50] # 36, 38, 42, 43规划失败， 33成功。
#     # index_range = [44, 47, 48, 49, 50] # 44, 47成功.
#     # index_range = [50] # 失败.
#     index_range = [44]
#     for index in index_range:
#         domain_name = 'blocksworld'
#         plan_detail = 'plan_detail.txt'
#         project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#         domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
#         problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")
#
#         model = parse_model(domain_file, problem_file)
#         domain = model[DOMAIN]
#         init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#         goal = model[INSTANCE][GOAL]
#         obj = model[INSTANCE][OBJECT]
#
#         plan, planner_time = solve_sub_goals_dw_v2(
#             init_state,
#             goal,
#             domain,
#             obj,
#             write_instance_to_remote,
#             planner,
#             30,
#             'blocksworld',
#             'BLOCKS',
#             index
#         )
#         # print([(action.action_name, action.parameters) for action in plan]) if plan != 'fail' else print(plan)

# # 大模型启发结果
# if __name__ == '__main__':
#     domain_name = 'blocksworld'
#     index = 1305
#     project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#     domain_file = os.path.join(project_dir, f"pddl_generator/{domain_name}/domain.pddl")
#     problem_file = os.path.join(project_dir, f"pddl_generator/{domain_name}/instances_2/instance-{index}.pddl")
#
#     model = parse_model(domain_file, problem_file)
#
#     domain = model[DOMAIN]
#     init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#     goal = model[INSTANCE][GOAL]
#     obj = model[INSTANCE][OBJECT]
#     # plan = ask4plan(domain, obj, init_state, goal, 1)
#     # print(plan[0])
#     # print(type(plan[0]))
#
#     action = ask4action(init_state, goal, [], applicable_actions(init_state, domain, obj))
#     print(action, type(action))


# 仅拆解目标
# if __name__ == '__main__':
#     domain_name = 'blocksworld'
#     project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#     domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
#     plan_detail = 'plan_detail.txt'
#     csv_file = 'plan_results.csv' # 记录结果的csv文件路径
#
#     for index in range(59, 101):
#         problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")
#         model = parse_model(domain_file, problem_file)
#         domain = model[DOMAIN]
#         goal = model[INSTANCE][GOAL]
#         pred = model[PREDICATES]
#         init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#
#         sorted_goal = sort_blocksworld_goals(goal) # 得到顺序执行的目标状态序列，这里仅有blocksworld领域需要调用
#         print(sorted_goal)
#         ins_list = get_ins_list(sorted_goal) # 子目标序列表
#         print(ins_list)
#
#         # 调用规划器DW
#         final_actions, all_time = solve_sub_goals_dw(
#             ins_list,
#             init_state,
#             domain,
#             write_instance_to_remote,
#             planner,
#             180
#         )
#
#         # 调用VAL审核，并且将具体规划时间和计划长度写入文件
#         val_and_write(csv_file, final_actions, index, all_time)


# DW本体规划
if __name__ == '__main__':
    for index in range(1, 2):
        domain_name = 'blocksworld'
        plan_detail = 'plan_detail.txt'
        csv_file = 'plan_results.csv' # 记录结果的csv文件路径
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
        problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")

        model = parse_model(domain_file, problem_file)
        goal = model[INSTANCE][GOAL]
        init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))

        print(init_state)
        print(goal)

        # final_actions, all_time = solve_goal_dw(
        #     init_state,
        #     goal,
        #     write_instance_to_remote,
        #     planner,
        #     180
        # )
        #
        # val_and_write(csv_file, final_actions, index, all_time)