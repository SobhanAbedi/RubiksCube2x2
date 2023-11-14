import numpy as np
import argparse
import time
from typing import List
from state import solved_state, next_state
from location import solved_location, next_location
from algo import solve


def quick_solve(method: str, testcase: int):
    state = solved_state()
    location = solved_location()
    f = open(f'testcases/{testcase}.txt', 'r')
    scramble_sequence = list(map(int, f.readline().split()))
    for a in scramble_sequence:
        state = next_state(state, action=a)
        location = next_location(location, action=a)
    print(f'SOLVING TESTCASE {testcase} WITH {method}...')
    start_time = time.time()
    solve_sequence = solve(state, location, method)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('actions:', solve_sequence)
    print(f'SOLVE FINISHED In {elapsed_time:.5f}s.\n')


def main_solve(args):
    # initializing state
    state = solved_state()
    location = solved_location()

    # scramble
    if args.testcase is None:
        scramble_sequence = np.random.randint(1, 12 + 1, np.random.randint(10, 30))
    else:
        f = open(args.testcase, 'r')
        scramble_sequence = list(map(int, f.readline().split()))

    # calculate the state and location
    for a in scramble_sequence:
        state = next_state(state, action=a)
        location = next_location(location, action=a)

    # solve rubik
    print('------------------ START ------------------')
    print('SOLVING...')
    start_time = time.time()
    solve_sequence = solve(state, location, method=args.method)
    print('actions:', solve_sequence)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'SOLVE FINISHED In {elapsed_time:.5f}s.')
    if len(solve_sequence) == 0:
        quit()
    print('--------- PRESS ENTER TO VISUALIZE --------')
    input()
    return scramble_sequence, solve_sequence


def single_method_time(method: str, tries: int = 100):
    slvd_state = solved_state()
    slvd_location = solved_location()
    total_elapsed_time = 0.0
    print(f'Measuring {method} average time for {tries} random cubes...')
    for i in range(tries):
        state = slvd_state
        location = slvd_location
        scramble_sequence = np.random.randint(1, 12 + 1, np.random.randint(10, 30))
        for a in scramble_sequence:
            state = next_state(state, action=a)
            location = next_location(location, action=a)
        start_time = time.time()
        solve(state, location, method, silent=True)
        end_time = time.time()
        total_elapsed_time += end_time - start_time
    average_time: float = total_elapsed_time / tries
    print(f'Average solve time is {average_time: .5f}s.')


def full_method_time():
    for i in range(1, 5):
        quick_solve('IDS-DFS', i)
    for i in range(1, 6):
        quick_solve('A*', i)
    for i in range(1, 8):
        quick_solve('BiBFS', i)
    for i in range(1, 8):
        quick_solve('BiA*', i)
    single_method_time("BiBFS")
    single_method_time("BiA*")


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--timetest', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--testcase', type=str, default=None)
    parser.add_argument('--method', type=str, default='Random')
    solve_args = parser.parse_args()

    global_scramble_sequence: List[int] = []
    global_solve_sequence: List[int] = []
    if not solve_args.manual and not solve_args.timetest:
        global_scramble_sequence, global_solve_sequence = main_solve(solve_args)
    if solve_args.timetest:
        if solve_args.method == 'Random':
            full_method_time()
        else:
            single_method_time(solve_args.method)
        quit()

    # imports
    from ursina import *
    from rubik import Rubik

    # start game
    app = Ursina(size=(1280, 720))
    rubik = Rubik()

    if solve_args.manual:
        rubik.text = Text('Manual', scale=2, origin=rubik.text_position)
        input = lambda key: rubik.action(key, animation_time=0.5)
    else:
        action_dict = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                       7: 'q', 8: 'w', 9: 'e', 10: 'r', 11: 't', 12: 'y'}

        # perform scramble + solution
        global_scramble_sequence = [action_dict[i] for i in global_scramble_sequence]
        global_solve_sequence = [action_dict[i] for i in global_solve_sequence]
        invoke(rubik.action_sequence, global_scramble_sequence, global_solve_sequence, delay=3.0)

    app.run()
