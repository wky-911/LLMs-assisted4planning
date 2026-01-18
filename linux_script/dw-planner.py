#!/usr/bin/env python3
import subprocess
import sys
import os


def run_planner(instance_number):
    # 检查是否传入了参数
    if instance_number is None:
        print("请传入一个参数 x，例如：./dw-planner.py 1")
        return

    # 定义要执行的命令
    command = [
        # "./fast-downward.py",
        # "instances/blocksworld/domain.pddl",
        # f"instances/blocksworld/instances/instance-{instance_number}.pddl",
        "./fast-downward.py",
        "blocksworld/domain.pddl",
        f"blocksworld/instances_2/instance-{instance_number}.pddl",
        "--search",
        "astar(lmcut())"
    ]

    # 执行命令
    try:
        # 使用subprocess.run执行命令
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=100)
        print("命令执行成功！")
        print("输出内容：")
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("命令执行超时，已强制终止！（超过180秒）")
        return
    except subprocess.CalledProcessError as e:
        print("命令执行失败！")
        print("错误信息：")
        print(e.stderr)
        return

    # 检查sas_plan文件是否存在
    plan_file = "sas_plan"
    if os.path.exists(plan_file):
        print("\n读取sas_plan文件内容：")
        with open(plan_file, "r") as file:
            plan_content = file.read()
            print(plan_content)
    else:
        print(f"未找到{plan_file}文件！")


if __name__ == "__main__":
    # 获取命令行参数
    instance_number = sys.argv[1] if len(sys.argv) > 1 else None
    run_planner(instance_number)