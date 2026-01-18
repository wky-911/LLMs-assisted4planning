import paramiko
import pandas as pd
from sympy.physics.units import action
from openai import OpenAI

from val.validate import validate_plan
from collections import defaultdict, deque
from components.llm_mystery import *

ds_key = os.getenv('DS_API_KEY')
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
directory='mystery'
domain='mystery-strips'


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

    def get_instance(self, instance_file=None, domain_name=domain, instance_name="temp_instance"):
        """
        根据当前状态(cur)和目标状态(goal)生成PDDL实例字符串。
        如果提供instance_file，则读取该实例模板文件，保留对象类型定义等模板结构，
        用当前对象的cur和goal替换其中的初始状态和目标状态。
        """
        # 如果提供了实例模板文件，则按照模板生成实例
        if instance_file is not None:
            with open(instance_file, 'r') as f:
                template = f.read()
            i_init = template.find("(:init")
            i_goal = template.find("(:goal")
            if i_init == -1 or i_goal == -1:
                raise ValueError("实例文件缺少:init或:goal部分")
            # 保留模板开头(初始状态之前的部分)和结尾(目标状态之后的部分)
            prefix = template[:i_init]
            # 查找目标状态块的结束位置
            count = 0
            goal_close_index = None
            for idx in range(i_goal, len(template)):
                char = template[idx]
                if char == '(':
                    count += 1
                elif char == ')':
                    if count == 0:
                        # 定义的额外右括号（在目标状态块之外）
                        goal_close_index = idx - 1
                        break
                    count -= 1
                    if count == 0:
                        goal_close_index = idx
                        break
            if goal_close_index is None:
                goal_close_index = template.rfind(')')
            suffix = template[goal_close_index + 1:]

            # 将当前状态和目标状态格式化为字符串
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
                    return str(pred)

            # 构建新的实例字符串
            result = prefix
            if not result.endswith('\n'):
                result += '\n'
            # 初始状态
            result += "(:init\n"
            for pred in self.cur:
                result += "        " + pred_to_str(pred) + "\n"
            result += ")\n\n"
            # 目标状态
            # 判断目标状态是单一谓词还是多个谓词
            if isinstance(self.goal, str) or (
                    isinstance(self.goal, (tuple, list)) and len(self.goal) > 0 and isinstance(self.goal[0], str)):
                goal_preds = [pred_to_str(self.goal)]
            else:
                goal_preds = [pred_to_str(pred) for pred in self.goal]
            if len(goal_preds) > 1:
                result += "(:goal (and\n"
                for gp in goal_preds:
                    result += "                " + gp + "\n"
                result += "        )\n"
                result += ")"  # 关闭:goal（不换行，与后面的定义关闭括号一起）
            else:
                result += "(:goal\n"
                result += "        " + goal_preds[0] + "\n"
                result += ")"
            # 添加模板剩余的部分（包括问题定义的闭合括号等）
            result += suffix
            return result
        else:
            # 未提供模板文件，使用原有方式生成（适用于无类型对象的领域）
            # 1. 自动提取所有涉及的对象
            objects = sorted({arg for pred in self.cur
                              if isinstance(pred, (tuple, list)) and len(pred) == 2
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
                    return str(pred)

            init_preds = [pred_to_str(pred) for pred in self.cur]
            # 目标状态可能是单一谓词或多个谓词
            if isinstance(self.goal, str) or (
                    isinstance(self.goal, (tuple, list)) and len(self.goal) > 0 and isinstance(self.goal[0], str)):
                goal_preds = [pred_to_str(self.goal)]
            else:
                goal_preds = [pred_to_str(pred) for pred in self.goal]
            # 3. 拼接PDDL实例字符串
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


def sort_mystery_goals(goal_list):
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

    # 2. 拓扑排序（和原来一样）
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

    # # 3. 按照顺序在前面插入 ontable 目标
    # result = [ ['ontable', [bottom]] for bottom in bottoms ] + result

    return result


# 得到子实例序列
def get_ins_list(sorted_goal):
    """
    输入实例的目标状态，获取子目标的顺序实现列表
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
    将实例写入ubuntu系统中，用于DW规划
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
        print("指令失败!!", error_text)
        ssh.close()
        return 'fail', None
    else:
        plan_marker = "读取 sas_plan 文件内容："
        plan_fail_marker = "Search stopped without finding a solution."
        planner_time_marker = "INFO     Planner time: "
        print(output_text)
        print(plan_marker in output_text)
        # 处理规划时间
        if planner_time_marker in output_text:
            planner_time_part = output_text.split(planner_time_marker, 1)[1].strip()
            planner_time = planner_time_part.split('\n', 1)[0]
            print(f"规划时间: {planner_time}")
        else:
            print("未找到规划时间信息")
            planner_time = None
        # 处理错误实例
        if plan_fail_marker in output_text:
            return 'this instance has no solution', planner_time
        # 处理规划结果
        if plan_marker in output_text:
            plan_part = output_text.split(plan_marker, 1)[1].strip()
            plan_lines = plan_part.splitlines()
            cleaned_plan = "\n".join(plan_lines[:-1])
        else:
            print("读取 sas_plan 文件异常, 超时规划！")
            ssh.close()
            return 'fail', None
    ssh.close()
    return cleaned_plan, planner_time


#调用DW解决实例，限定时间为10s
def solve_goal_dw(problem_file, init_state, goal_state, write_instance_to_remote, planner, timeout=10, directory=directory, domain_name=domain):
    instance = InsChild(goal_state)
    instance.set_cur(init_state)
    instance_temp = instance.get_instance(problem_file, domain_name)
    write_instance_to_remote(instance_temp)
    plan, planner_time = planner(timeout, directory)
    if plan == 'fail':
        print("plan fail!")
        return plan, None
    elif plan == 'this instance has no solution':
        return plan, planner_time
    else:
        print('plan success!')
        instance.set_actions(plan)
        instance.set_time(planner_time)
        actions = instance.get_actions()
        return actions, planner_time


# 调用DW解决所有的子实例
def solve_sub_goals_dw(problem_file, ins_list, init_state, domain, write_instance_to_remote, planner, timeout=30, directory=directory, domain_name=domain):
    """
    批量依次求解每个子目标，自动传递状态，并合并所有动作为最终计划
    :param ins_list: InsList对象，已按顺序装好所有子目标
    :param init_state: 初始状态（list/tuple/集合）
    :param domain: 领域描述（parse_model结果里的domain）
    :param write_instance_to_remote: 写文件方法
    :param planner: 远程调用规划器方法
    :return: total_actions（最终完整动作序列list[Action]）
    """
    cur_state = init_state
    total_actions = []
    for i in range(len(ins_list.ins_children)):
        child = ins_list.get_children(i)
        child.set_cur(cur_state)  # 将最新状态设置为当前子目标的初始状态
        temp_instance = child.get_instance(problem_file, domain_name)
        write_instance_to_remote(temp_instance)
        plan, planner_time = planner(timeout, directory)
        print(f"单目标规划成功，时长为: {planner_time}") if planner_time else None
        #TODO 解决超时返回空字符串
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


def val_and_write(csv_file, final_actions, index, all_time, call_count=0):
    # 判断表格文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 定义要写入的一行内容（每次一个实例）
    if final_actions == 'fail':
        result_row = {
            "实例序号": index,
            "规划时间": all_time,
            "计划步骤数": "fail",
            "调用大模型次数": call_count,
            "领域": directory
        }
    elif final_actions == 'this instance has no solution':
        result_row = {
            "实例序号": index,
            "规划时间": all_time,
            "计划步骤数": "no solution",
            "调用大模型次数": call_count,
            "领域": directory
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
                "计划步骤数": "fail",
                "调用大模型次数": call_count,
                "领域": directory
            }

    # 写入csv（如果没有表头先写表头，否则直接追加）
    df = pd.DataFrame([result_row])
    if not file_exists:
        df.to_csv(csv_file, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(csv_file, index=False, mode='a', header=False, encoding='utf-8-sig')


# DW本体规划
if __name__ == '__main__':
    for index in range(7, 8):
        domain_name = directory

        plan_detail = 'plan_detail.txt'  # 记录结果的csv文件路径
        csv_file = 'plan_results.csv'
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
        problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")

        model = parse_model(domain_file, problem_file)
        goal = model[INSTANCE][GOAL]
        init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))

        print(problem_file)
        print(init_state)
        print(goal)

        # final_actions, all_time = solve_goal_dw(
        #     problem_file,
        #     init_state,
        #     goal,
        #     write_instance_to_remote,
        #     planner,
        #     180
        # )
        # print(f'final actions :{final_actions}, all time :{all_time}')
        # val_and_write(csv_file, final_actions, index, all_time)

