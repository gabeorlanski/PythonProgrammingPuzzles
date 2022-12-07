import copy
from datetime import datetime
import functools
import itertools
from pathlib import Path
import json
import multiprocessing as mp
from collections import defaultdict
from typing import List
import ast
import random
import math
import numpy as np

random.seed(100)
puzzle_path = Path(
    "/usr/local/google/home/gabeorlanski/PythonProgrammingPuzzles/fix_puzzles/puzzles.json"
)

fixes = {
    "BoxVolume:1": "def sol(options=[2, 512, 1024], n=340282366920938463463374607431768211456, max_dim=13):"
}

DEFAULT_HEADER = """from typing import List"""
MAX_STR_LEN = 10000
MAX_ITERABLE_ELEMENTS = 5000
MAX_NUM_LENGTH = 19
DEBUG_NAMES = [
    # "study.Study_1",
    # "study.Study_10",
    # "study.Study_11",
    # "study.Study_12",
    # "study.Study_13",
    # "study.Study_14",
    # "study.Study_15",
    # "study.Study_16",
    # "study.Study_17",
    # "study.Study_18",
    # "study.Study_19",
    # "study.Study_2",
    # "study.Study_20",
    # "study.Study_21",
    # "study.Study_22",
    # "study.Study_23",
    # "study.Study_24",
    # "study.Study_25",
    # "study.Study_26",
    # "study.Study_27",
    # "study.Study_28",
    # "study.Study_29",
    # "study.Study_3",
    # "study.Study_30",
    # "study.Study_4",
    # "study.Study_5",
    # "study.Study_6",
]


def obj_to_annot(v):
    def get_sub_types(children):
        out = [
            t for t in map(obj_to_annot, children) if t and t not in ["Dict", "List"]
        ]
        if len(set(out)) > 1:
            raise ValueError(f"Too Many types {list(children)}, {v=}")
        return out

    if not isinstance(v, (list, dict)):
        return type(v).__name__

    if isinstance(v, dict):
        key_types = get_sub_types(v.keys())
        value_types = get_sub_types(v.values())
        if not key_types or not value_types:
            return "Dict"
        return f"Dict[{key_types[0]},{value_types[0]}]"

    child_type = get_sub_types(v)
    try:
        child_type = child_type[0]
    except IndexError as e:
        return "List"

    return f"List[{child_type}]"


def too_big_num(v):
    if isinstance(v, int):
        return len(str(abs(v))) > MAX_NUM_LENGTH
    elif isinstance(v, float):
        return len(str(abs(v))) > MAX_NUM_LENGTH
    elif not isinstance(v, (dict, tuple, set, list)):
        return False

    if isinstance(v, (list, tuple, set)):
        return any(map(too_big_num, v))
    return any(map(too_big_num, v.values()))


def get_args_and_types(code):
    tree = ast.parse(code).body[0]
    f_args = tree.args
    default_values = f_args.defaults
    if len(f_args.args) > len(default_values):
        default_values = [None] * (
            len(f_args.args) - len(default_values)
        ) + default_values

    for i in range(len(default_values)):
        if default_values[i] is not None:
            default_values[i] = ast.literal_eval(ast.unparse(default_values[i]))

    args = []
    values = []
    for arg, default_value in zip(f_args.args, default_values):
        annotation = arg.annotation
        if annotation is None and default_value is not None:
            annotation = obj_to_annot(default_value)
        elif annotation is not None:
            annotation = ast.unparse(annotation)

        args.append([arg.arg, annotation])
        values.append(default_value)

    return args, values


def nested_length(v):
    if not isinstance(v, (list, set, tuple, dict)):
        return 1

    out = 0
    if isinstance(v, (list, set, tuple)):
        for c in v:
            out += nested_length(c) + 1
    else:
        for c in v.values():
            out += nested_length(c) + 1
    return out


def get_questions(ignored_questions):
    input_types = defaultdict(set)
    out_input_types = defaultdict(list)
    question_groups = {}

    skipped_to_long = set()
    skipped_to_many_elements = set()
    skipped_to_large_num = set()
    for problem in json.load(puzzle_path.open()):
        # if print_limit == 0:
        #     break
        # print_limit -= 1
        if not problem["sol_bodies"]:
            continue
        exe_code = "\n".join(problem["sol_bodies"])
        if problem["name"] in fixes:
            header = fixes[problem["name"]]
        else:
            header = problem["sol_header"]
        exe_code = f"{DEFAULT_HEADER}\n{header}\n{exe_code}\nRESULT=sol()"

        name, idx = problem["name"].split(":", 1)
        name = f'{problem["module"]}.{name}'
        if name in ignored_questions:
            print(f"Skipping {name} because it is ignored.")
            continue
        if DEBUG_NAMES and name not in DEBUG_NAMES:
            continue

        if name not in question_groups:
            question_groups[name] = []
        print(f"Executing: {name} {idx=}")
        # print(exe_code)
        try:
            exec(exe_code)
        except Exception as e:
            print(exe_code)
            raise e

        result = locals()["RESULT"]
        if not isinstance(problem["sat"], str):
            print(problem["sat"])
            continue

        exec(f'{problem["sat"]}\nIS_TRUE=sat(result)')
        if not locals()["IS_TRUE"]:
            print(f'{problem["name"]} did not pass sat.')
        try:
            args, values = get_args_and_types(problem["sat"])
        except Exception as e:
            print(problem["sat"])
            raise e

        if (
            isinstance(result, (tuple, list, dict))
            and nested_length(result) >= MAX_ITERABLE_ELEMENTS
        ):
            print(
                f'Skipping {problem["module"]}-{problem["name"]} because it had too large of a iterable'
            )
            skipped_to_many_elements.add(name)
            continue

        if too_big_num(result):
            print(
                f'Skipping {problem["module"]}-{problem["name"]} because it had too large number.'
            )
            skipped_to_large_num.add(name)
            continue

        if not isinstance(result, tuple):
            result = (result,)
        for i, v in enumerate(result):
            assert values[i] is None, problem["sat"]
            values[i] = v

        # print('\t'+problem['sat'].split('\n')[0])
        # print(f'\ttypes:')
        assert problem["ans_type"] == args[0][1]

        arg_dict = {}
        arg_order = []
        for n, v in args:
            arg_order.append(n)
            arg_dict[n] = v
            # print(f'\t\t{n} = {v}')

        if any(len(str(v)) > 10000 for v in values):
            skipped_to_long.add(name)
            print(f'{problem["name"]} had a value that was too long.')
            continue
        # print('\targs:')
        correct_vals = {}
        for i, v in enumerate(values):
            input_types[args[i][1]].add(str(v) if not isinstance(v, str) else repr(v))
            correct_vals[args[i][0]] = v
            # print(f'\t\t{args[i][0]} = {repr(v)[:25]}')
        question_groups[name].append(
            {
                "name": name,
                "input_sig": arg_dict,
                "input_order": arg_order,
                "correct": correct_vals,
                "code": problem["sat"],
                "docstring": problem["sol_docstring"],
                "tags": problem["tags"],
                "weight": problem["weight"],
            }
        )

    for k in input_types:
        new_types = []
        for v in sorted(input_types[k], key=len):
            try:
                new_types.append(ast.literal_eval(v))
            except SyntaxError as e:
                continue
        out_input_types[k] = new_types
    print(f"{len(question_groups)} total questions")
    print(f"{len(skipped_to_long)} skipped for length.")
    print(f"{len(skipped_to_many_elements)} skipped too many elements.")

    print(f"{len(skipped_to_large_num)} skipped too many elements.")
    empty_group = [k for k, v in question_groups.items() if len(v) == 0]
    if empty_group:
        print(f"{len(empty_group)} did not have any values")

    failures = {
        "too_many_elements": list(skipped_to_many_elements),
        "too_long": list(skipped_to_long),
        "to_big_num": list(skipped_to_large_num),
        "empty": empty_group,
    }
    return question_groups, out_input_types, failures


class RemoveAssertions(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert) -> ast.If:
        new_node = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=node.test),
            body=[ast.Return(value=ast.Constant(False))],
            orelse=[],
        )
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        new_node = copy.deepcopy(node)
        new_node.args.defaults = []
        return ast.copy_location(new_node, node)


def execute_test_input(args):
    idx, program, test_inputs = args
    arg_inputs = json.dumps(test_inputs)

    code = f"{program}\nRESULT=sat(**json.loads('{arg_inputs}'))"
    result = None
    # print(code)
    try:
        start = datetime.now()
        exec(code)
    except AssertionError as e:
        result = "AssertionError"
    except Exception as e:
        return (idx, type(e).__name__, None)
    runtime = (datetime.now() - start).total_seconds()
    if result is None:
        result = locals()["RESULT"]
    return idx, str(result), runtime


def get_empty(t, default_val):

    if t == "str":
        return ""
    if t.startswith("Dict"):
        return {}
    if t.startswith("List"):
        return []
    return default_val


def get_all_empty_possible(q):

    out = []
    for k, v in q["input_sig"].items():
        empty = get_empty(v, q["correct"][k])
        if not isinstance(empty, (int, float)) and empty != q["correct"][k]:
            new_dict = copy.deepcopy(q["correct"])
            new_dict[k] = empty
            out.append(new_dict)
    return out


def exec_new_tests(current_tests, current_false_tests):
    results = []
    has_false = set()
    false_tests = 0
    input_mapping = {v[0]: v[-1] for v in current_tests}
    with mp.Pool(6, maxtasksperchild=4) as pool:
        for result in pool.imap(execute_test_input, current_tests):
            results.append(result)

            if result[1] == "False" or result[1] == "AssertionError":
                has_false.add(result[0][0])
                false_tests += 1
            if len(results) % 5000 == 0:
                print(
                    f"Finished {len(results)}/{len(current_tests)}. {len(has_false)} Questions with a false test. {false_tests} total false tests."
                )

        # Cleanup pool
        pool.close()
        pool.terminate()
    for k, r, runtime in results:
        if r == "False" or r == "AssertionError":
            current_false_tests[k[0]].append((runtime, input_mapping[k]))
    return current_false_tests


def get_possible_inputs(signature, invalid_values, input_values, offsets):
    out = []
    for n, t in signature.items():
        potential = input_values[t]
        offset = offsets[n]
        for new_value in potential[offset + 1 :]:

            offsets[n] += 1
            if new_value not in invalid_values[n]:
                other_args = [k for k in invalid_values if k != n]
                product_arrs = [invalid_values[k] for k in other_args]
                for prod in itertools.product(*product_arrs):

                    input_dict = {k: v for k, v in zip(other_args, prod)}
                    input_dict[n] = new_value
                    out.append(input_dict)

                break

    return offsets, out


def random_nested_list(rng, dim, sample_values=None, bounds=100):
    size = rng.choice(np.arange(0, 15), size=1)[0]

    if dim == 0:
        if size == 0:
            return []
        if sample_values is not None:
            sampled = rng.choice(sample_values, size=size, replace=True)
        else:
            sampled = rng.uniform(bounds * -1, bounds, size=size)
        return sampled.tolist()
    else:
        out = []
        for _ in range(size):
            out.append(random_nested_list(rng, dim - 1, sample_values, bounds))
        return out


def main():
    ignored = [
        "study.Study_8",
        "classic_puzzles.Kirkman",
        "basic.EngineerNumbers",
        "IMO.FindRepeats",
        "games.Mastermind",
        "games.TicTacToeO",
        "trivial_inverse.ListIndex2",
        "tutorial.Tutorial4",
        "human_eval.ZobristCollision",
        "games.TicTacToeX",
        "probability.BirthdayParadoxMonteCarlo",
    ]
    manual_tests = {
        "study.Study_27": [{"li": [1, 2, 3, 4]}, {"li": [1, 1, 1, 2]}],
        "study.Study_16": [{"s": "1.23"}, {"s": "4.51"}],
        "study.Study_21": [
            {"li": list(range(15))},
            {"li": [1, 2, 3, 1, 3, 3, 1, 2, 3, 1, 2, 3]},
        ],
        "study.Study_30": [{"li": list(range(30, 1, -1))}, {"li": list(range(22))}],
        "classic_puzzles.Quine": [
            {"quine": "(lambda x: chr(34))"},
            {"quine": "(lambda x: 'Hello World')"},
        ],
        "classic_puzzles.RevQuine": [{"rev_quine": "'no'"}, {"rev_quine": "'yessir'"}],
        "classic_puzzles.SquaringTheSquare": [
            {"xy_sides": [[1, 2, 3]]},
            {"xy_sides": [[1, 50, 2]]},
        ],
        "games.ZeroSum": [
            dict(
                strategies=[[0.3, 0.2, 0.5]] * 2,
                A=[[0.0, -0.5, 1.0], [0.75, 0.0, -1.0], [-1.0, 0.4, 0.0]],
                eps=0.01,
            )
        ],
        "human_eval.SevenElevenThirteen": [dict(li=[[71, 0]], n=3, lower=1)],
        "human_eval.Fibonacci": [
            dict(nums=[1, 3], n=2),
            dict(nums=[1, 1, 2, 3, 6], n=5),
        ],
        "human_eval.CumulativeSums": [
            dict(sums=[0, 1, 5], n=2),
            dict(sums=[0, 1, 3, 6, 7], n=4),
        ],
        "number_theory.CollatzDelay":[dict(n=-1,t=10,upper=10),dict(n=10,t=3,upper=4)]
    }
    rng = np.random.default_rng(100)
    question_groups, input_types, failures = get_questions(ignored)
    input_types["List[int]"].extend([[23, 23, 96], []])
    for k in ignored:
        print(f"Removing {k}")
        question_groups.pop(k, None)
    no_tests = []
    for k in list(question_groups.keys()):
        if len(question_groups[k]) == 0:
            del question_groups[k]
            no_tests.append(k)
    false_tests = {k: [] for k in question_groups}
    offsets = {k: {} for k in question_groups}
    all_inputs = {k: {} for k in question_groups}
    current_tests = []

    num_rand_to_add = 50
    input_types["float"].extend(rng.uniform(-100, 100, size=num_rand_to_add).tolist())
    input_types["int"].extend(
        [int(rng.uniform(-250, 250)) for _ in range(num_rand_to_add)]
    )

    print(f"{len(input_types)} input types found.")
    TOTAL_TESTS = 0
    for k in input_types:
        print(f"\t{k} = {len(input_types[k])}")
    for k, q_list in question_groups.items():
        new_tests_for_q = get_all_empty_possible(q_list[0])
        code = q_list[0]["code"]
        for i, t in enumerate(new_tests_for_q):
            current_tests.append(((k, TOTAL_TESTS), code, t))
            TOTAL_TESTS += 1

        input_order = q_list[0]["input_order"]
        offsets[k] = {a: 0 for a in input_order}
        q_all_inputs = {a: set() for a in input_order}
        for q in q_list:
            for a in input_order:
                cur_val = q["correct"][a]
                q_all_inputs[a].add(
                    str(cur_val) if not isinstance(cur_val, str) else repr(cur_val)
                )
        q_converted_values = {a: [] for a in input_order}
        for a, val_set in q_all_inputs.items():
            converted = []
            for question_list in val_set:
                try:
                    converted_value = ast.literal_eval(question_list)
                except:
                    print(question_list)
                    continue
                q_converted_values[a].append(converted_value)
                converted.append(q_converted_values)

            all_inputs[k][a] = converted

        if len(q_list) == 1:
            # print(f"{k} only has 1 test")
            continue

        for i, v_list in enumerate(
            itertools.product(*list(q_converted_values.values()))
        ):
            current_tests.append(
                ((k, TOTAL_TESTS), code, {k: v for k, v in zip(input_order, v_list)})
            )
            TOTAL_TESTS += 1
    print("Failures:")
    for k, question_list in failures.items():
        print(f"\t{k} = {question_list}")

    false_tests = exec_new_tests(current_tests, false_tests)
    print(f"Executed {len(current_tests)}")
    print(f"{sum(map(len,false_tests.values()))} false tests found")

    for k, tests in manual_tests.items():
        false_tests[k].extend([(0, t) for t in tests])
        print(
            f"Added {len(tests)} false test(s) to {k}, total is now {len(false_tests[k])}"
        )

    needs_more_tests = list(
        filter(lambda k: len(false_tests[k]) == 0, false_tests.keys())
    )
    print(f"{len(needs_more_tests)} Still need false tests")
    other_input_tests = []

    for k in input_types:
        input_types[k] = sorted(input_types[k], key=lambda v: len(str(v)))

    num_to_gen = 5
    for k in needs_more_tests:
        q_list = question_groups[k]
        code = q_list[0]["code"]
        k_offsets = offsets[k]
        for _ in range(num_to_gen):
            k_offsets, new_tests = get_possible_inputs(
                q_list[0]["input_sig"], all_inputs[k], input_types, k_offsets
            )
            for i, test in enumerate(new_tests):

                other_input_tests.append(((k, TOTAL_TESTS), code, test))
                TOTAL_TESTS += 1

    print(f"{len(other_input_tests)} new tests")
    false_tests = exec_new_tests(other_input_tests, false_tests)
    print(f"{sum(map(len,false_tests.values()))} false tests found")
    needs_more_tests = list(
        filter(
            lambda k: len(false_tests[k]) == 0 and k not in manual_tests,
            false_tests.keys(),
        )
    )

    all_questions = []
    max_false_tests = 2
    for k, question_list in question_groups.items():
        docstring = question_list[0]["docstring"].replace('"""', "")
        new_docstring = []
        tags = set(t for q in question_list for t in q["tags"])

        for l in docstring.split("\n"):
            if not l.strip():
                continue
            new_docstring.append(l.lstrip())

        docstring = "\n".join(new_docstring)
        code = question_list[0]["code"].lstrip()
        tree = RemoveAssertions().visit(ast.parse(code))

        qid = str(len(all_questions))

        test_inputs = [(t["correct"], True) for t in question_list]

        false_tests_for_k = list(
            sorted(
                false_tests[k],
                key=lambda x: sum(map(lambda y: len(str(y)), x[1].values())),
            )
        )[: max_false_tests * 2]
        false_tests_for_k = list(sorted(false_tests_for_k, key=lambda x: x[0]))[
            :max_false_tests
        ]
        for test in false_tests_for_k:
            test_inputs.append((test[1], False))

        input_order = question_list[0]["input_order"]
        testing_code = []
        for test, output in test_inputs:
            test_argument = []
            for arg in input_order:
                cleaned_test_arg = (
                    repr(test[arg])
                    if not isinstance(test[arg], str)
                    else repr(test[arg])
                )
                test_argument.append(cleaned_test_arg.replace("\n", "\\n"))

            testing_code.append(f'assert sat({", ".join(test_argument)}) == {output}')
        problem_dict = {
            "id": qid,
            "text": f"Verifies that the inputs satisfy the problem:\n{docstring}",
            "solution": ast.unparse(tree),
            "title": f"TP3/{k.replace('.py','')}",
            "testing_code": "\n".join(testing_code),
            "entry_fn_name": "sat",
            "metadata": {
                "tags": list(tags),
                "weight": question_list[0]["weight"],
            },
        }
        all_questions.append(problem_dict)

    print(f"{needs_more_tests} have no false tests.")
    print(f"{no_tests} had no tests.")
    with open("tp3_questions.jsonl", "w") as f:
        for l in all_questions:
            f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
    print("Starting")
    main()
