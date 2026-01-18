import sys
import tarski
import tarski.io
from tarski.io.fstrips import print_init, print_goal, print_formula, print_atom
from tarski.syntax import CompoundFormula, formulas, Tautology, Atom
from tarski.syntax.terms import CompoundTerm, Constant
from tarski.syntax.sorts import Interval
from tarski.fstrips import AddEffect, DelEffect
from tarski.fstrips.fstrips import FunctionalEffect, IncreaseEffect

# from constants import *
from .constants import *
# from writer_new import ModelWriter
# from .writer_new import ModelWriter


"""
这个脚本的目的是将 PDDL/FSTRIPS 文件转换为一个内部模型字典，
用于后续 MCTS / 规划算法的处理。

本版本在原始基础上做了增强：
- 递归解析 goal / precondition / 条件效应中的公式，支持 folding2 等复杂 domain
- 避免对 CompoundFormula/QuantifiedFormula 误用 .symbol 导致 AttributeError
"""


# ========= 辅助工具函数：从任意公式中递归收集 Atom =========

def _collect_atoms(formula, out_list):
    """
    从 Tarski 的公式对象中递归收集所有 Atom。
    当前版本不区分正/负极性，只是把 Atom 都抽出来，避免解析崩溃。

    :param formula: 可能是 Atom、CompoundFormula、QuantifiedFormula 等
    :param out_list: list，用来累积 Atom
    """
    if formula is None:
        return

    # 1. 原子公式
    if isinstance(formula, Atom):
        out_list.append(formula)

    # 2. 带 subformulas 的复合公式（and / or / not / implication 等）
    elif hasattr(formula, 'subformulas'):
        try:
            for sf in formula.subformulas:
                _collect_atoms(sf, out_list)
        except TypeError:
            # 某些实现 subformulas 可能不是可迭代；防御性处理
            pass

    # 3. 其他类型（量词、数值表达式等），暂时不展开，保持“不崩溃优先”
    else:
        # 如果以后需要支持 forall/exists，可以在这里按需要展开
        pass


# ========= 顶层入口 =========

def parse_model(domain_file, problem_file):
    reader = tarski.io.FstripsReader()
    reader.read_problem(domain_file, problem_file)
    model_dict = store_model(reader)
    return model_dict


def store_model(reader):
    model_dict = {}
    model_dict[METRIC] = reader.problem.plan_metric
    model_dict[PREDICATES] = store_predicates(reader)
    model_dict[FUNCTIONS] = store_functions(reader)

    model_dict[INSTANCE] = {}
    model_dict[INSTANCE][INIT] = {}
    model_dict[INSTANCE][INIT][FUNCTIONS], model_dict[INSTANCE][INIT][PREDICATES] = store_init(reader)
    model_dict[INSTANCE][GOAL] = store_goal(reader)
    model_dict[INSTANCE][OBJECT] = store_objects(reader)

    model_dict[DOMAIN] = store_actions(reader)

    model_dict[HIERARCHY] = {}
    model_dict[HIERARCHY][ANCESTORS], model_dict[HIERARCHY][IMM_PARENT] = store_hierarchy(reader)

    model_dict[CONSTANTS] = store_constants(reader)
    return model_dict


# ========= 语言层信息 =========

def store_predicates(reader):
    predicates = list(reader.problem.language.predicates)
    predicates_list = []
    for preds in predicates:
        # 过滤内置比较运算符
        if str(preds.symbol) in ['=', '!=', '<', '<=', '>', '>=']:
            continue
        predicates_list.append(
            [preds.symbol, [sorts.name for sorts in preds.sort]]
        )
    return predicates_list


def store_constants(reader):
    constants = reader.problem.language.constants()
    constant_list = []
    for constant in constants:
        constant_list.append([constant.symbol, constant.sort.name])
    return constant_list


def store_functions(reader):
    functions = list(reader.problem.language.functions)
    functions_list = []
    for funcs in functions:
        # 过滤通用算子与内置函数
        if str(funcs.symbol) in ['ite', '@', '+', '-', '*', '/', '**', '%', 'sqrt', 'number']:
            continue
        functions_list.append(
            [funcs.symbol, [sorts.name for sorts in funcs.sort]]
        )
    return functions_list


# ========= 初始状态 =========

def store_init(reader):
    inits = reader.problem.init.as_atoms()
    init_dict = {}
    init_dict[FUNCTIONS] = []
    init_dict[PREDICATES] = []

    for i in range(len(inits)):
        atom = inits[i]
        # 非 Atom：通常是函数赋值形式 [ (f ...), value ]
        if not isinstance(atom, Atom):
            init_dict[FUNCTIONS].append(
                [atom[0].symbol.symbol, [atom[1].symbol]]
            )
        else:
            # 纯命题
            if len(atom.subterms) == 0:
                init_dict[PREDICATES].append(
                    [atom.symbol.symbol, []]
                )
            # 带参数的谓词
            else:
                init_dict[PREDICATES].append(
                    [atom.symbol.symbol, [subt.symbol for subt in atom.subterms]]
                )

    return init_dict[FUNCTIONS], init_dict[PREDICATES]


# ========= 目标 =========

def store_goal(reader):
    """
    将 Tarski 的 goal 公式转为内部列表形式：
    [[predicate_name, [arg1, arg2, ...]], ...]
    支持嵌套 AND/NOT 等复合公式，只提取其中的所有 Atom。
    """
    goal = reader.problem.goal
    goals = []

    # 目标恒真：没有实际目标，返回空列表
    if isinstance(goal, Tautology):
        return goals

    def collect(formula):
        if isinstance(formula, Atom):
            goals.append(
                [formula.symbol.symbol, [subt.symbol for subt in formula.subterms]]
            )
        elif hasattr(formula, 'subformulas'):
            for sf in formula.subformulas:
                collect(sf)
        else:
            # 量词、其他复杂公式暂不展开
            pass

    collect(goal)
    return goals


# ========= 动作定义 =========

def store_actions(reader):
    action_model = {}

    for act in reader.problem.actions.values():
        action_model[act.name] = {}

        # 参数列表
        action_model[act.name][PARARMETERS] = [
            (p.symbol, p.sort.name) for p in act.parameters
        ]

        # ---------- 前置条件：递归收集所有 Atom ----------
        pos_prec_atoms = []
        _collect_atoms(act.precondition, pos_prec_atoms)

        action_model[act.name][POS_PREC] = [
            [atom.symbol.symbol, [t.symbol for t in atom.subterms]]
            for atom in pos_prec_atoms
        ]

        # 目前 NEG_PREC 暂未填充（需要区分 NOT 的极性时再扩展）
        # action_model[act.name][NEG_PREC] = [...]

        # ---------- 效应 ----------
        action_model[act.name][ADDS] = []
        action_model[act.name][DELS] = []
        action_model[act.name][FUNCTIONAL] = []
        action_model[act.name][COND_ADDS] = []
        action_model[act.name][COND_DELS] = []
        action_model[act.name][COST] = act.cost

        for curr_effs in act.effects:
            # 某些规划器可能把多个效应打包成 list，这里统一转为 list
            if not isinstance(curr_effs, list):
                curr_effs = [curr_effs]

            for eff in curr_effs:
                # ---------- 有条件效应 ----------
                if not isinstance(eff.condition, Tautology):
                    # 解析条件里的公式：递归抽出所有 Atom
                    cond_atoms = []
                    _collect_atoms(eff.condition, cond_atoms)

                    # 保持原来形状： [ [ [pred,[args]], ... ] ]
                    curr_condition = [[
                        [a.symbol.symbol, [t.symbol for t in a.subterms]]
                        for a in cond_atoms
                    ]]

                    # 条件 Add / Del
                    if isinstance(eff, AddEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][COND_ADDS].append(
                                [curr_condition, [eff.atom.symbol.symbol, []]]
                            )
                        else:
                            action_model[act.name][COND_ADDS].append(
                                [curr_condition,
                                 [eff.atom.symbol.symbol,
                                  [subt.symbol for subt in eff.atom.subterms]]]
                            )

                    elif isinstance(eff, DelEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][COND_DELS].append(
                                [curr_condition, [eff.atom.symbol.symbol, []]]
                            )
                        else:
                            action_model[act.name][COND_DELS].append(
                                [curr_condition,
                                 [eff.atom.symbol.symbol,
                                  [subt.symbol for subt in eff.atom.subterms]]]
                            )

                    # 条件功能效应（数值）
                    elif isinstance(eff, FunctionalEffect):
                        lhs = eff.lhs      # 左侧函数项
                        rhs = eff.rhs      # 右侧表达式/常量

                        # 右侧可能是 CompoundTerm / Constant / 其他
                        if isinstance(rhs, CompoundTerm):
                            rhs_repr = [rhs.symbol.symbol, rhs.sort.name]
                        elif isinstance(rhs, Constant):
                            rhs_repr = [rhs.symbol, rhs.sort.name]
                        else:
                            rhs_repr = [str(rhs),
                                        getattr(getattr(rhs, 'sort', None), 'name', None)]

                        action_model[act.name][FUNCTIONAL].append(
                            [[lhs.symbol.symbol, lhs.sort.name], rhs_repr]
                        )

                # ---------- 无条件效应 ----------
                else:
                    if isinstance(eff, AddEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][ADDS].append(
                                [eff.atom.symbol.symbol, []]
                            )
                        else:
                            action_model[act.name][ADDS].append(
                                [eff.atom.symbol.symbol,
                                 [subt.symbol for subt in eff.atom.subterms]]
                            )

                    elif isinstance(eff, DelEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][DELS].append(
                                [eff.atom.symbol.symbol, []]
                            )
                        else:
                            action_model[act.name][DELS].append(
                                [eff.atom.symbol.symbol,
                                 [subt.symbol for subt in eff.atom.subterms]]
                            )

                    elif isinstance(eff, FunctionalEffect):
                        # 无条件功能效应（increase/assign 等）
                        lhs = eff.lhs
                        rhs = eff.rhs
                        if isinstance(rhs, CompoundTerm):
                            rhs_repr = [rhs.symbol.symbol, rhs.sort.name]
                        elif isinstance(rhs, Constant):
                            rhs_repr = [rhs.symbol, rhs.sort.name]
                        else:
                            rhs_repr = [str(rhs),
                                        getattr(getattr(rhs, 'sort', None), 'name', None)]

                        action_model[act.name][FUNCTIONAL].append(
                            [[lhs.symbol.symbol, lhs.sort.name], rhs_repr]
                        )

    return action_model


# ========= 类型层次结构 =========

def store_hierarchy(reader):
    ancestors = reader.problem.language.ancestor_sorts
    ancestor_list = []
    for key, value in ancestors.items():
        # 所有 sort 的祖先列表
        ancestor_list.append(
            [key.name, [v.name for v in value], int(isinstance(key, Interval))]
        )

    imm_parents = reader.problem.language.immediate_parent
    imm_parent_list = []
    for key, value in imm_parents.items():
        if getattr(value, 'name', None) is None:
            imm_parent_list.append(
                [key.name, None, int(isinstance(key, Interval))]
            )
        else:
            imm_parent_list.append(
                [key.name, value.name, int(isinstance(key, Interval))]
            )
    return ancestor_list, imm_parent_list


# ========= 对象 =========

def store_objects(reader):
    # 对于大多数 PDDL/FSTRIPS 编码，对象与常量统一在 language.constants 里
    objects = reader.problem.language.constants()
    object_list = [obj.symbol for obj in objects]
    return object_list


# ========= 测试入口（可按需修改） =========

if __name__ == '__main__':
    # 示例：你可以改成 folding2 的 domain/problem 文件路径做本地测试
    # model = parse_model('folding2-domain.pddl', 'folding2-problem.pddl')
    model = parse_model('domain_folding.pddl', 'instance-11.pddl')
    print(model)
    goal = model[INSTANCE][GOAL]
    init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
    print(goal)
    print(init_state)

