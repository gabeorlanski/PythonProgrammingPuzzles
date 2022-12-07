import generators
import json


# study.Study_16
# study.Study_21
# study.Study_27
# study.Study_30
# classic_puzzles.Quine
# classic_puzzles.RevQuine
# classic_puzzles.SquaringTheSquare


question_set = "number_theory"
question = "CollatzDelay"

gen_cls = getattr(getattr(generators, question_set), question)
gen = gen_cls()


tests = [dict(n=-1,t=10,upper=10),dict(n=10,t=3,upper=4)]

for t in tests:
    print(f'{t=} ==> {gen_cls.sat(**t)}')

# print(gen_cls.sat()
