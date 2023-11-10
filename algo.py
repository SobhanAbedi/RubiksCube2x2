import copy
from dataclasses import dataclass, field
from collections import deque
from queue import PriorityQueue
import numpy as np
import numpy.typing as npt
from typing import List, Dict
from state import next_state, solved_state
from location import next_location, location_diff

SolvedState = solved_state()
SolvedStateHash = hash(SolvedState.data.tobytes())
LocationManhattanDist = location_diff()
MaxMoves = 9


@dataclass
class GraphNode:
    state: npt.NDArray
    path: List
    move_count: int


@dataclass(order=True)
class GraphNodeAStar:
    utility: int
    removed: bool = field(compare=False)
    state: npt.NDArray = field(compare=False)
    location: npt.NDArray = field(compare=False)
    path: List = field(compare=False)
    cost: int = field(compare=False)
    heuristic: int = field(compare=False)


@dataclass
class NodeSeen:
    cost: int
    node: GraphNodeAStar | None


def ids_dfs(init_state: npt.NDArray) -> List:
    first_node = GraphNode(init_state, [], 0)
    for limit in range(1, MaxMoves + 1):
        print(f'limit = {limit}')
        fringe = deque['GraphNode']()
        fringe.appendleft(first_node)
        # smaller than zero means expanded and larger than zero means still in the fringe
        seen: Dict = {first_node.state.tobytes(): 0}
        explored = 0
        expanded = 1
        while len(fringe) > 0:
            explored += 1
            cur_node: GraphNode = fringe.popleft()
            cur_node.move_count = seen[cur_node.state.tobytes()]
            seen[cur_node.state.tobytes()] = -cur_node.move_count
            if SolvedStateHash == hash(cur_node.state.data.tobytes()):
                print(f'{expanded} Nodes Expanded')
                print(f'{explored} Nodes Explored')
                return cur_node.path
            if cur_node.move_count < limit:
                nxt_node_move_count = cur_node.move_count + 1
                act_to_pass = 0
                if len(cur_node.path) > 0:
                    act_to_pass = ((cur_node.path[-1] + 5) % 12) + 1
                for act in range(1, 13):
                    if act == act_to_pass:
                        continue
                    nxt_node_state = next_state(cur_node.state, act)
                    key = nxt_node_state.tobytes()
                    if key in seen:
                        val = seen[key]
                        if val >= 0 or -val <= nxt_node_move_count:
                            continue
                    seen[key] = nxt_node_move_count
                    nxt_node_path = copy.copy(cur_node.path)
                    nxt_node_path.append(act)
                    nxt_node = GraphNode(nxt_node_state, nxt_node_path, nxt_node_move_count)
                    fringe.appendleft(nxt_node)
                    expanded += 1
                del cur_node
        print(f'{expanded} Nodes Expanded')
        print(f'{explored} Nodes Explored')
    print(f'No answers found with upto {MaxMoves} moves')
    return []


def heuristic(location: npt.NDArray) -> int:
    lin_loc = location.reshape(8)
    total_dist4 = 0
    for i in range(8):
        total_dist4 += LocationManhattanDist[i, lin_loc[i] - 1]
    return total_dist4


def a_star(init_state: npt.NDArray, init_location: npt.NDArray) -> List:
    h = heuristic(init_location)
    first_node = GraphNodeAStar(h, False, init_state, init_location, [], 0, h)
    fringe = PriorityQueue['GraphNodeAStar']()
    fringe.put(first_node)
    seen: Dict[bytes, NodeSeen] = {first_node.state.tobytes(): NodeSeen(0, first_node)}
    explored = 0
    expanded = 1
    while not fringe.empty():
        cur_node = fringe.get()
        while cur_node.removed and not fringe.empty():
            del cur_node
            cur_node = fringe.get()
        if cur_node.removed:  # meaning last while ended because fringe was empty
            break
        key = cur_node.state.tobytes()
        node_seen = seen[key]
        node_seen.cost *= -1
        node_seen.node = None
        explored += 1
        if SolvedStateHash == hash(cur_node.state.data.tobytes()):
            print(f'{expanded} Nodes Expanded')
            print(f'{explored} Nodes Explored')
            return cur_node.path
        nxt_node_cost = cur_node.cost + 4
        act_to_pass = 0
        if len(cur_node.path) > 0:
            act_to_pass = ((cur_node.path[-1] + 5) % 12) + 1
        for act in range(1, 13):
            if act == act_to_pass:
                continue
            nxt_node_state = next_state(cur_node.state, act)
            nxt_node_location = next_location(cur_node.location, act)
            nxt_node_path = copy.copy(cur_node.path)
            nxt_node_path.append(act)
            key = nxt_node_state.tobytes()
            if key in seen:
                node_seen: NodeSeen = seen[key]
                if node_seen.cost > 0 and node_seen.cost > nxt_node_cost:
                    # if node_seen.node is None:  # could be removed for production run
                    #     print("!!!Unexpected Error!!!")
                    #     quit()
                    node_seen.cost = nxt_node_cost
                    node_seen.node.removed = True
                    nxt_node = GraphNodeAStar(node_seen.node.heuristic + nxt_node_cost, False, nxt_node_state,
                                              nxt_node_location, nxt_node_path, nxt_node_cost, node_seen.node.heuristic)
                    node_seen.node = nxt_node
                    fringe.put(nxt_node)
                    expanded += 1
                    continue
                elif node_seen.cost == 0 or -node_seen.cost <= nxt_node_cost:
                    continue
            nxt_node_heuristic = heuristic(nxt_node_location)
            nxt_node = GraphNodeAStar(nxt_node_heuristic + nxt_node_cost, False, nxt_node_state,
                                      nxt_node_location, nxt_node_path, nxt_node_cost, nxt_node_heuristic)
            seen[key] = NodeSeen(nxt_node_cost, nxt_node)
            fringe.put(nxt_node)
            expanded += 1
        del cur_node
    print(f'{expanded} Nodes Expanded')
    print(f'{explored} Nodes Explored')
    print(f'No answers found')
    return []


def bi_bfs(init_state: npt.NDArray) -> List:
    first_node = GraphNode(init_state, [], 0)
    final_node = GraphNode(SolvedState, [], 0)
    fringe = [deque['GraphNode'](), deque['GraphNode']()]
    fringe[0].append(first_node)
    fringe[1].append(final_node)
    # True means expanded and False means still in the fringe
    seen: List[Dict[bytes, List[int]]] = [{first_node.state.tobytes(): []}, {final_node.state.tobytes(): []}]
    explored = 0
    expanded = 2
    turn: int = 1
    not_turn: int = 0
    while True:
        not_turn = turn
        turn = (turn + 1) % 2
        nxt_layer_number = fringe[turn][0].move_count + 1
        while fringe[turn][0].move_count < nxt_layer_number:
            cur_node = fringe[turn].popleft()
            explored += 1
            key = cur_node.state.tobytes()
            if key in seen[not_turn]:
                print(f'{expanded} Nodes Expanded')
                print(f'{explored} Nodes Explored')
                frst: List[int] = []
                scnd: List[int] = []
                if turn == 0:
                    frst = cur_node.path
                    scnd = seen[1][key]
                else:
                    frst = seen[0][key]
                    scnd = cur_node.path
                scnd.reverse()
                for i in range(len(scnd)):
                    scnd[i] = ((scnd[i] + 5) % 12) + 1
                return frst + scnd
            act_to_pass = 0
            if len(cur_node.path) > 0:
                act_to_pass = ((cur_node.path[-1] + 5) % 12) + 1
            for act in range(1, 13):
                if act == act_to_pass:
                    continue
                nxt_node_state = next_state(cur_node.state, act)
                key = nxt_node_state.tobytes()
                if key in seen[turn]:
                    continue
                nxt_node_path = copy.copy(cur_node.path)
                nxt_node_path.append(act)
                seen[turn][key] = nxt_node_path
                nxt_node = GraphNode(nxt_node_state, nxt_node_path, nxt_layer_number)
                fringe[turn].append(nxt_node)
                expanded += 1
            del cur_node


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.
 
    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == 'Random':
        return list(np.random.randint(1, 12 + 1, 10))

    elif method == 'IDS-DFS':
        return ids_dfs(init_state)

    elif method == 'A*':
        return a_star(init_state, init_location)

    elif method == 'BiBFS':
        return bi_bfs(init_state)

    else:
        return []
