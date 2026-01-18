#!/usr/bin/env python3
import subprocess
import sys
import os

def run_planner(timeout, directory):
    # 构建指令参数
    import os
    fast_downward_path = os.path.abspath("./fast-downward.py")
    domain_path = os.path.abspath(f"{directory}/domain.pddl")
    instance_path = os.path.abspath(f"{directory}/instance_temp.pddl")
    print("fast_downward_path:", fast_downward_path)
    print("domain_path:", domain_path)
    print("instance_path:", instance_path)
    print("cwd:", os.getcwd())
    command = [
        fast_downward_path,
        domain_path,
        instance_path,
        "--search",
        "astar(lmcut())"
    ]
    print(f"command: {command}")
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        print("命令执行成功！")
        print("输出内容：")
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("命令执行超时，已强制终止！")
        return
    except subprocess.CalledProcessError as e:
        print(f"command: {command}")
        print("命令执行失败！")
        print("错误信息：")
        print(e.stderr or e.output or e.stdout or "无详细错误")
        return

    # 检查 sas_plan 文件是否存在
    plan_file = "sas_plan"
    if os.path.exists(plan_file):
        print("\n读取 sas_plan 文件内容：")
        with open(plan_file, "r") as file:
            plan_content = file.read()
            print(plan_content)
    else:
        print(f"未找到 {plan_file} 文件！")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <timeout> <directory>")
        sys.exit(1)

    timeout = int(sys.argv[1])
    directory = sys.argv[2]
    run_planner(timeout, directory)
