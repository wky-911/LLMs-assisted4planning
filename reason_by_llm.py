import yaml
import pandas as pd
from pathlib import Path
from openai import OpenAI
from val.validate import validate_plan

from components.model_parser.parser_new import *
from pddl import parse_domain, parse_problem
from pddl.logic.functions import EqualTo as PddlEqualTo
from pddl.logic.predicates import Predicate as PddlPredicate
from pddl.logic.functions import NumericFunction as PddlNumericFunction, NumericValue as PddlNumericValue


ds_key = os.getenv('DS_API_KEY')
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

# client = OpenAI(
#     api_key=os.environ["METACHAT_API_KEY"],
#     base_url="https://llm-api.mmchat.xyz/v1"
# )

# model_name = "gpt-5"
# plan_txt = "gpt_plan.txt"
# record_txt = "gpt_record.csv"
# model_name = "claude-sonnet-4-20250514"
# plan_txt = "claude_plan.txt"
# record_txt = "claude_record.csv"
model_name = "deepseek-reasoner"
plan_txt = "ds_plan.txt"
record_txt = "ds_record.csv"

def _term_name(t):
    return getattr(t, "name", str(t))


def literal_to_atom_str(lit):
    """
    把一个 *正* 的 Predicate 变成规范字符串: (pred a b)
    """
    name = getattr(lit, "name", None) or getattr(lit, "predicate", None) or str(lit)
    terms = getattr(lit, "terms", getattr(lit, "args", []))
    args = [_term_name(x) for x in terms]
    if args:
        return f"({name} {' '.join(args)})"
    return f"({name})"


def parse_init_to_state(init_facts):
    """
    输入 problem.init（一般是 frozenset），抽取：
    - atoms: set[str]，每个元素形如 '(pred a b)'
    - numeric: dict[str,float]，例如 total-cost / rotate-cost / update-cost
    """
    atoms = set()
    numeric = {}

    for item in init_facts:
        if isinstance(item, PddlPredicate):
            atoms.add(literal_to_atom_str(item))
            continue

        # (= (total-cost) 0) 在 pddl0.4.3 是 pddl.logic.functions.EqualTo
        if isinstance(item, PddlEqualTo):
            left, right = item.operands
            if isinstance(left, PddlNumericFunction):
                fname = left.name
                # folding2 里的 cost 函数一般是 0-arity
                if isinstance(right, PddlNumericValue):
                    numeric[fname] = float(right.value)
                else:
                    numeric[fname] = float(getattr(right, "value", right))
            continue

    numeric.setdefault("total-cost", 0.0)
    numeric.setdefault("rotate-cost", 1.0)
    numeric.setdefault("update-cost", 0.0)
    return {"atoms": atoms, "numeric": numeric}


def ask_llm(client, prompt4sys, prompt4usr):
    completion = client.chat.completions.create(
        # model="gpt-4.1",
        model=model_name,
        # messages=[
        #     {
        #         "role": "user",
        #         "content": "Write a one-sentence bedtime story about a unicorn."
        #     }
        # ]
        messages=[
            {"role": "system", "content": prompt4sys},
            {"role": "user", "content": prompt4usr},
        ]
    )

    raw_output = completion.choices[0].message.content
    with open(plan_txt, 'w', encoding='utf-8') as f:
        f.write(raw_output)


def val_ans(domain_name, index, csv_file):
    domain_file = f"ipc_instances/domain_{domain_name}.pddl"
    problem_file = f"ipc_instances/{domain_name}/instance-{index}.pddl"
    plan_file = Path(plan_txt)
    # VAL审核
    result = validate_plan(
        domain_path=domain_file,
        problem_path=problem_file,
        plan_path=plan_file,
    )

    plan_len = sum(1 for line in plan_file.open(encoding='utf-8') if line.strip())
    if result["is_valid"]:
        result_row = {
            "实例序号": index,
            "计划步骤数": plan_len,
            "领域": domain_name
        }
    else:
        result_row = {
            "实例序号": index,
            "计划步骤数": "fail",
            "领域": domain_name
        }
    df = pd.DataFrame([result_row])
    df.to_csv(csv_file, index=False, mode='a', header=False)


# def val_ans4folding(index, csv_file):
#
#     # 用folding_validator进行验证
#     ok, report = FoldingValidator().validate(
#         "ipc_instances/domain_folding.pddl",
#         f"ipc_instances/folding/instance-{index}.pddl",
#         plan_txt,
#     )
#     print(report)
#
#     plan_len = sum(1 for line in Path(plan_txt).open(encoding='utf-8') if line.strip())
#     if ok:
#         result_row = {
#             "实例序号": index,
#             "计划步骤数": plan_len,
#             "领域": 'folding'
#         }
#     else:
#         result_row = {
#             "实例序号": index,
#             "计划步骤数": "fail",
#             "领域": 'folding'
#         }
#     df = pd.DataFrame([result_row])
#     df.to_csv(csv_file, index=False, mode='a', header=False)


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

    with open(plan_txt, 'w', encoding='utf-8') as f:
        f.write(raw_output)


# # 针对Folding领域
# if __name__ == '__main__':
#     csv_file = record_txt
#     # with open('configs/prompt4other.yaml', 'r', encoding='utf-8') as file:
#     with open('configs/prompt4ds.yaml', 'r', encoding='utf-8') as file:
#         sys_data = yaml.safe_load(file)
#     domain_file = f"ipc_instances/domain_folding.pddl"
#     prompt4sys = sys_data.get('folding', '')
#
#     for index in range(1, 2):
#         problem_file = f"ipc_instances/folding/instance-{index}.pddl"
#         problem = parse_problem(problem_file)
#         domain_ast = parse_domain(domain_file)
#
#         goal = problem.goal
#         actions = [action.name for action in domain_ast.actions]
#         cur_state = parse_init_to_state(problem.init)
#         init_state = [state for state in cur_state.get('atoms')]
#         # prompt4usr = f"The goal state: {g oal} \n"
#         # prompt4usr += f"The init state: {init_state} \n"
#         # prompt4usr += f"The actions: {actions}, and the returned action names must be consistent\n"
#         # prompt4usr += f"Only return the plan without any other information"
#         prompt4usr = f"目标状态: {goal} \n"
#         prompt4usr += f"初始状态: {init_state} \n"
#         prompt4usr += f"动作集: {actions}，返回的动作名一定要保持一致\n"
#         prompt4usr += f"只返回计划，不用返回其他任何信息"
#         print(prompt4sys)
#         print(prompt4usr)
#         # ask_llm(client, prompt4sys, prompt4usr)
#         ask_ds(client, prompt4sys, prompt4usr)
#         val_ans4folding(index, csv_file)


if __name__ == '__main__':
    # directory_info = [('blocksworld', 51), ('logistics', 43), ('gripper', 21), ('depot', 23), ('mystery', 31)]
    # directory_info = [('logistics', 43), ('gripper', 21), ('depot', 23), ('mystery', 31)]
    # directory_info = [('depot', 23), ('mystery', 31)]
    # directory_info = [('logistics', 43), ('depot', 23)]
    directory_info = [('blocksworld', 2)]
    csv_file = record_txt
    with open('configs/prompt4other.yaml', 'r', encoding='utf-8') as file:
        sys_data = yaml.safe_load(file)
    for (domain_name, instance_num) in directory_info:
        domain_file = f"ipc_instances/domain_{domain_name}.pddl"
        prompt4sys = sys_data.get(domain_name, '')
        # TODO 起始id
        for index in range(1, instance_num):
            problem_file = f"ipc_instances/{domain_name}/instance-{index}.pddl"
            model = parse_model(domain_file, problem_file)
            domain = model[DOMAIN]
            actions = [action for action in domain.keys()]
            init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
            goal = model[INSTANCE][GOAL]
            print(f"{domain_name}: {index}")
            print(init_state)
            print(goal)
            print('-' * 20)
            prompt4usr = f"The goal state: {goal} \n"
            prompt4usr += f"The init state: {init_state} \n"
            prompt4usr += f"The actions: {actions}, and the returned action names must be consistent\n"
            prompt4usr += f"Only return the plan without any other information"
            print(prompt4sys)
            print(prompt4usr)
            ask_llm(client, prompt4sys, prompt4usr)
            val_ans(domain_name, index, csv_file)
