from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Iterable
from dataclasses import dataclass
from components.llm_blocksworld import *

"""
依赖总图构建 成环检测 以及解耦条件生成
"""

directory='blocksworld'
# directory='depot'
Pred = Tuple[str, Tuple[str, ...]]
GoalItem = List

@dataclass
class PredGraph:
    edges: Dict[str, List[str]]
    rev: Dict[str, List[str]]
    nodes: Set[str]
    in_deg: Dict[str, int]
    out_deg: Dict[str, int]
    edge_label: Dict[Tuple[str,str], str]  # (u,v) -> predicate name that created this edge

    def bottoms(self) -> List[str]:
        return [n for n in self.nodes if self.in_deg.get(n, 0) == 0]

    def successors(self, u: str) -> List[str]:
        return self.edges.get(u, [])

    def predecessors(self, v: str) -> List[str]:
        return self.rev.get(v, [])

def _ensure_node(n: str, nodes: Set[str], in_deg: Dict[str,int], out_deg: Dict[str,int]):
    if n not in nodes:
        nodes.add(n)
    if n not in in_deg:
        in_deg[n] = 0
    if n not in out_deg:
        out_deg[n] = 0

def _goal_predicate_names(goal: List[GoalItem]) -> Set[str]:
    names: Set[str] = set()
    def visit(x):
        if isinstance(x, list):
            if len(x) >= 1 and isinstance(x[0], str):
                head = x[0].lower()
                if head == 'not':
                    if len(x) >= 2:
                        visit(x[1])
                else:
                    names.add(head)
                    for y in x[1:]:
                        visit(y)
            else:
                for y in x:
                    visit(y)
    visit(goal)
    return names

def _apply_pred_semantics(pname: str, args: Tuple[str, ...],
                          edges: Dict[str, List[str]],
                          rev: Dict[str, List[str]],
                          nodes: Set[str],
                          in_deg: Dict[str,int], out_deg: Dict[str,int],
                          edge_label: Dict[Tuple[str,str], str]):
    """
    Default semantics (BlocksWorld-like):
      - on(x,y): add edge y->x and label it 'on'
      - ontable(x): register node
      - others: register nodes only (easy to extend later)
    """
    if pname == 'on' and len(args) == 2:
        x, y = args[0], args[1]
        _ensure_node(x, nodes, in_deg, out_deg)
        _ensure_node(y, nodes, in_deg, out_deg)
        edges[y].append(x)
        rev[x].append(y)
        in_deg[x] += 1
        out_deg[y] += 1
        edge_label[(y, x)] = pname
    elif pname == 'ontable' and len(args) == 1:
        _ensure_node(args[0], nodes, in_deg, out_deg)
    else:
        for a in args:
            _ensure_node(a, nodes, in_deg, out_deg)

def build_goal_pred_graph(goal: List[GoalItem]) -> PredGraph:
    edges: Dict[str, List[str]] = defaultdict(list)
    rev: Dict[str, List[str]] = defaultdict(list)
    nodes: Set[str] = set()
    in_deg: Dict[str,int] = {}
    out_deg: Dict[str,int] = {}
    labels: Dict[Tuple[str,str], str] = {}

    for item in goal:
        if not item:
            continue
        head = item[0].lower()
        if head == 'not':
            inner = item[1] if len(item) >= 2 else None
            if isinstance(inner, list) and len(inner) >= 1:
                pname = inner[0].lower()
                args = inner[1] if len(inner) >= 2 else []
                if isinstance(args, list):
                    args = tuple(args)
                _apply_pred_semantics(pname, args, edges, rev, nodes, in_deg, out_deg, labels)
        else:
            pname = head
            args = item[1] if len(item) >= 2 else []
            if isinstance(args, list):
                args = tuple(args)
            _apply_pred_semantics(pname, args, edges, rev, nodes, in_deg, out_deg, labels)
    return PredGraph(dict(edges), dict(rev), nodes, in_deg, out_deg, labels)

def build_init_pred_graph(init_state: List[Pred],
                          goal: List[GoalItem] = None,
                          goal_pred_names: Set[str] = None) -> PredGraph:
    """
    Auto-detect predicate names from goal if provided; else fallback to {'on','ontable'}.
    """
    if goal_pred_names is None:
        if goal is not None:
            goal_pred_names = _goal_predicate_names(goal)
        else:
            goal_pred_names = {'on', 'ontable'}

    edges: Dict[str, List[str]] = defaultdict(list)
    rev: Dict[str, List[str]] = defaultdict(list)
    nodes: Set[str] = set()
    in_deg: Dict[str,int] = {}
    out_deg: Dict[str,int] = {}
    labels: Dict[Tuple[str,str], str] = {}

    for pred, args in init_state:
        pname = pred.lower()
        if pname not in goal_pred_names:
            continue
        _apply_pred_semantics(pname, args, edges, rev, nodes, in_deg, out_deg, labels)

    return PredGraph(dict(edges), dict(rev), nodes, in_deg, out_deg, labels)

def topo_on_order(goal: List[GoalItem]) -> List[GoalItem]:
    """
    For now, still orders 'on' relations bottom-up. Extend if you add other binary stacking preds.
    """
    edges: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = defaultdict(int)
    nodes: Set[str] = set()
    on_dict: Dict[Tuple[str,str], GoalItem] = {}

    def add_on(x: str, y: str, original_item: GoalItem):
        edges[y].append(x)
        indeg[x] += 1
        nodes.add(x); nodes.add(y)
        on_dict[(x,y)] = original_item

    for item in goal:
        if not item:
            continue
        head = item[0].lower()
        if head == 'not':
            inner = item[1] if len(item) >= 2 else None
            if isinstance(inner, list) and len(inner) >= 2 and inner[0].lower() == 'on':
                args = inner[1]
                if isinstance(args, list) and len(args) == 2:
                    x, y = args[0], args[1]
                    add_on(x, y, inner)
        else:
            if head == 'on' and len(item) >= 2:
                args = item[1]
                if isinstance(args, list) and len(args) == 2:
                    x, y = args[0], args[1]
                    add_on(x, y, item)

    for n in list(nodes):
        indeg.setdefault(n, 0)

    q = deque([n for n in nodes if indeg[n] == 0])
    result: List[GoalItem] = []
    while q:
        u = q.popleft()
        for v in edges.get(u, []):
            if (v, u) in on_dict:
                result.append(on_dict[(v,u)])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return result

def chain_from_bottom(g: PredGraph, bottom: str) -> List[str]:
    chain = [bottom]
    cur = bottom
    seen = {bottom}
    while True:
        succs = sorted(g.successors(cur))
        if not succs:
            break
        nxt = succs[0]
        if nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt
    return chain

def prefix_to_node(g: PredGraph, target: str) -> List[str]:
    path_rev = [target]
    cur = target
    seen = {target}
    while True:
        preds = sorted(g.predecessors(cur))
        if not preds:
            break
        prv = preds[0]
        if prv in seen:
            break
        path_rev.append(prv)
        seen.add(prv)
        cur = prv
    return list(reversed(path_rev))

def merge_prefix_and_goal(init_g: PredGraph, goal_g: PredGraph, deduplicate: bool = True) -> List[List[str]]:
    merged: List[List[str]] = []
    for a in sorted(goal_g.bottoms()):
        left = prefix_to_node(init_g, a)[:-1]
        right = chain_from_bottom(goal_g, a)
        if deduplicate:
            right_set = set(right)
            left = [x for x in left if x not in right_set]
        merged.append(left + right)
    return merged

def merge_with_anchors(init_g: PredGraph, goal_g: PredGraph, deduplicate: bool = True) -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []
    for a in sorted(goal_g.bottoms()):
        left = prefix_to_node(init_g, a)[:-1]
        right = chain_from_bottom(goal_g, a)
        if deduplicate:
            right_set = set(right)
            left = [x for x in left if x not in right_set]
        out.append((a, left + right))
    return out

def _edge_predicate_for_pair(b: str, X: str, goal_g: PredGraph, init_g: PredGraph) -> str:
    # Prefer the goal graph label (as cycles are often caused by right-chain edges),
    # fall back to init graph label, else default to 'on' for BlocksWorld.
    return goal_g.edge_label.get((b, X)) or init_g.edge_label.get((b, X)) or 'on'

def _first_cycle_preferring_head(chain: List[str]):
    if not chain:
        return None
    # 1) 头结点是否复现
    head = chain[0]
    for j in range(1, len(chain)):
        if chain[j] == head:
            return (head, 0, 1)  # b=head, i0=0, X=chain[1]

    # 2) 否则：从头扫描第一个重复节点
    first_index: Dict[str, int] = {}
    for i, node in enumerate(chain):
        if node in first_index:
            i0 = first_index[node]
            if i0 + 1 < len(chain):
                return (node, i0, i0 + 1)
            else:
                return None
        else:
            first_index[node] = i
    return None


def first_cycle_neg_goal_for_chain_auto(init_g: PredGraph,
                                        goal_g: PredGraph,
                                        chain: List[str],
                                        anchor_a: str) -> List[GoalItem]:
    """
    修改点：不再要求 anchor_a 在初始图中必须有前驱；即便无前驱，也执行“从头回返检测”。

    规则：
      - 优先检查 head 是否在后续再次出现：若出现，b=head，X=chain[1]；
      - 否则回退到从头扫描遇到的“第一个重复节点” b 及其首次出现处的直接后继 X；
      - 谓词 p 自动从边标签推断（优先 goal_g，再退 init_g，兜底 'on'）；
      - 若无环，返回 []。
    """
    if not chain:
        return []

    res = _first_cycle_preferring_head(chain)
    if res is None:
        return []
    b, i0, succ_idx = res
    X = chain[succ_idx]

    # 自动选择谓词（优先使用“目标图”标签，再退回“初始图”）
    p = goal_g.edge_label.get((b, X)) or init_g.edge_label.get((b, X)) or 'on'
    return ['not', [p, [X, b]]]


def cycle_anchors_for_merged(merged_with_anchors: List[Tuple[str, List[str]]]
                             ) -> List[Tuple[str, str]]:
    """
    对每条链返回 (cycle_node, chain_str)。若无环，则忽略该链（也可按需保留）。
    注意：这里只负责“把锚点改成成环对象”，真正是否成环的判断由 _first_cycle_preferring_head 完成。
    """
    out: List[Tuple[str, str]] = []
    for _, chain in merged_with_anchors:
        det = _first_cycle_preferring_head(chain)
        if det is not None:
            b, i0, succ_idx = det
            out.append((b, pretty_chain(chain)))
    return out


def suggest_neg_goals_for_merged_auto(init_g: PredGraph,
                                      goal_g: PredGraph,
                                      merged_with_anchors: List[Tuple[str, List[str]]]
                                      ) -> List[GoalItem]:
    """
    批量处理版本：对每条 (anchor, chain) 都做“从头回返检测”，
    不再检查 anchor 是否在初始图有前驱。
    """
    suggestions: List[GoalItem] = []
    for a, chain in merged_with_anchors:
        sug = first_cycle_neg_goal_for_chain_auto(init_g, goal_g, chain, a)
        if sug:
            suggestions.append(sug)
    return suggestions


def pretty_chain(chain: Iterable[str]) -> str:
    return "->".join(chain)


if __name__ == '__main__':
    index = 37
    domain_name = directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    domain_file = os.path.join(project_dir, f"ipc_instances/domain_{domain_name}.pddl")
    problem_file = os.path.join(project_dir, f"ipc_instances/{domain_name}/instance-{index}.pddl")
    model = parse_model(domain_file, problem_file)
    domain = model[DOMAIN]
    init_state = list(set(tuple([p[0], tuple(p[1])]) for p in model[INSTANCE][INIT][PREDICATES]))
    goal = model[INSTANCE][GOAL]
    obj = model[INSTANCE][OBJECT]

    goal_g = build_goal_pred_graph(goal)
    init_g = build_init_pred_graph(init_state, goal=goal)

    merged = merge_with_anchors(init_g, goal_g, deduplicate=False)

    print(f"id = {index}")
    print(f"init_state = {init_state}")
    print(f"goal = {goal}")
    print(f"obj = {obj}")
    print(cycle_anchors_for_merged(merged))
    print(suggest_neg_goals_for_merged_auto(init_g, goal_g, merged))