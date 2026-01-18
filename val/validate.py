import subprocess
import os

def validate_plan(domain_path, problem_path, plan_path):
    # 构建验证命令（包含详细输出和文件路径）
    command = ['Validate.exe', '-v', domain_path, problem_path, plan_path]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        # print(result.stdout)

        # 判断输出是否包含成功标志（根据实际输出调整关键词）
        is_valid = "Plan valid" in result.stdout
        return {
            "is_valid": is_valid,
            "output": result.stdout,
            "error": result.stderr
        }

    except subprocess.CalledProcessError as e:
        return {
            "is_valid": False,
            "output": e.stdout,
            "error": e.stderr
        }



# 处理blocksworld领域的结果
# 处理logistics领域的结果
# 处理elevator领域的结果
# 处理gripper领域的结果
# 处理depot领域的结果
# 处理mystery-1领域的结果


if __name__ == '__main__':
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(project_dir)

    # 调用示例（使用绝对路径）
    result = validate_plan(
        domain_path=os.path.join(project_dir, "val/Depots.pddl"),
        # domain_path=os.path.join(project_dir, "components/blocksworld/domain_elevator.pddl"),
        # domain_path=os.path.join(project_dir, "components/logistics/domain_elevator.pddl"),
        problem_path=os.path.join(project_dir, "val/pfile1.pddl"),
        # problem_path=os.path.join(project_dir, "components/blocksworld/problem_simple.pddl"),
        # problem_path=os.path.join(project_dir, "components/logistics/problem.pddl"),
        plan_path=os.path.join(project_dir, "val/pfile1.plan")
        # plan_path=os.path.join(project_dir, "components/result_bw.txt"),
        # plan_path=os.path.join(project_dir, "components/result_lt.txt")
    )
    print(result)
    print("计划是否有效:", result["is_valid"])
    print("验证输出:", result["output"])