import os
import yaml
import pandas as pd
from pathlib import Path
from openai import OpenAI
from components.model_parser.parser_new import *
from itertools import permutations
from val.validate import validate_plan

ds_key = os.getenv('DS_API_KEY')
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

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

    with open('ds_plan.txt', 'w', encoding='utf-8') as f:
        f.write(raw_output)


def val_ans(domain_name, index, csv_file):
    domain_file = f"ipc_instances/domain_{domain_name}.pddl"
    problem_file = f"ipc_instances/{domain_name}/instance-{index}.pddl"
    plan_file = Path('ds_plan.txt')
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
    df.to_csv(csv_file, index=False, mode='a', header=False, encoding='utf-8-sig')


if __name__ == '__main__':
    directory_info = [('blocksworld', 51), ('logistics', 43), ('gripper', 21), ('depot', 23), ('mystery', 31)]
    csv_file = 'ds_record.csv'
    with open('configs/prompt4ds.yaml', 'r', encoding='utf-8') as file:
        sys_data = yaml.safe_load(file)
    for (domain_name, instance_num) in directory_info:
        domain_file = f"ipc_instances/domain_{domain_name}.pddl"
        prompt4sys = sys_data.get(domain_name, '')
        print(prompt4sys) 
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
            prompt4usr = f"目标状态为： {goal} \n"
            prompt4usr += f"初始状态为： {init_state} \n"
            prompt4usr += f"动作名为： {actions}， 返回的动作名一定要保持一致\n"
            prompt4usr += f"只返回计划，不用返回其他任何信息"
            ask_ds(client, prompt4sys, prompt4usr)
            val_ans(domain_name, index, csv_file)
            # print(prompt)
            # with open(target_file, 'w', encoding='utf-8') as f:
            #     f.write(prompt)


# if __name__ == '__main__':
#     index = 1
#     domain_name = 'blocksworld'
#     # domain_name = 'logistics'
#     # domain_name = 'gripper'
#     # domain_name = 'depot'
#     # domain_name = 'mystery'
#     plan_file = 'ds_plan.txt'
#     csv_file = 'ds_record.csv'
#     val_ans(domain_name, index, csv_file)


# if __name__ == '__main__':
#     # domain_file = f"ipc_instances/domain_blocksworld.pddl"
#     # problem_file = f"ipc_instances/blocksworld/instance-11.pddl"
#     # model = parse_model(domain_file, problem_file)
#     # domain = model[DOMAIN]
#     # actions =  [action for action in domain.keys()]
#     # print(actions)
#     directory_info = [('blocksworld', 51), ('logistics', 85), ('gripper', 21), ('depot', 23), ('mystery', 31)]
#     with open('configs/prompt4ds.yaml', 'r', encoding='utf-8') as file:
#         sys_data = yaml.safe_load(file)
#     for (domain_name, instance_num) in directory_info:
#         domain_file = f"ipc_instances/domain_{domain_name}.pddl"
#         prompt4sys = sys_data.get(domain_name, '')
#         print(prompt4sys)
#         for index in range(1, instance_num):
#             problem_file = f"ipc_instances/{domain_name}/instance-{index}.pddl"
#             target_file = f"prompt4ds/{domain_name}/instance-{index}.txt"
#             model = parse_model(domain_file, problem_file)
#             domain = model[DOMAIN]
#             actions = [action for action in domain.keys()]
#             init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
#             goal = model[INSTANCE][GOAL]
#             print(f"{domain_name}: {index}")
#             print(init_state)
#             print(goal)
#             print('-'*20)
#             prompt = prompt4sys
#             prompt += f"目标状态为： {goal} \n"
#             prompt += f"初始状态为： {init_state} \n"
#             prompt += f"动作名为： {actions}， 返回的动作名一定要保持一致\n"
#             prompt += f"只返回计划，不用返回其他任何信息"
#             print(prompt)
#             with open(target_file, 'w', encoding='utf-8') as f:
#                 f.write(prompt)