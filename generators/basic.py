"""Problems testing basic knowledge -- easy to solve if you understand what is being asked"""

from puzzle_generator import MAX_DIGITS, PuzzleGenerator, Tags
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles

class SumOfDigits(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(x: str, s=274):
        """Find a number that its digits sum to a specific value."""
        return s == sum([int(d) for d in x])

    @staticmethod
    def sol(s):
        return int(s / 9) * '9' + str(s % 9)

    def gen_random(self):
        s = self.random.randint(0, 10 ** 5)
        self.add(dict(s=s))


class FloatWithDecimalValue(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(z: float, v=9, d=0.0001):
        """Create a float with a specific decimal."""
        return int(z * 1 / d % 10) == v

    @staticmethod
    def sol(v, d):
        return v * d

    def gen_random(self):
        v = self.random.randint(0, 9)
        a = self.random.randint(-MAX_DIGITS, MAX_DIGITS)
        while a == 0:
            a = self.random.randint(-MAX_DIGITS, MAX_DIGITS)
        d = float(10 ** a)
        if not float((v * d) * 1 / d % 10) == v:
            # Some values won't be solved by the reference solution due to Python floats.
            return
        self.add(dict(v=v, d=d))


class ArithmeticSequence(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], a=7, s=5, e=200):
        """Create a list that is a subrange of an arithmetic sequence."""
        return x[0] == a and x[-1] <= e and (x[-1] + s > e) and all([x[i] + s == x[i + 1] for i in range(len(x) - 1)])

    @staticmethod
    def sol(a, s, e):
        return list(range(a, e + 1, s))

    def gen_random(self):
        a = self.random.randint(-10 ** 5, 10 ** 1)
        e = self.random.randint(a, 10 ** 3)
        s = self.random.randint(1, 10)
        self.add(dict(a=a, e=e, s=s))


class GeometricSequence(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], a=8, r=2, l=10):
        """Create a list that is a subrange of an gemoetric sequence."""
        return x[0] == a and len(x) == l and all([x[i] * r == x[i + 1] for i in range(len(x) - 1)])

    @staticmethod
    def sol(a, r, l):
        return [a * r ** i for i in range(l)]

    def gen_random(self):
        a = self.random.randint(-10 ** 3, 10 ** 3)
        r = self.random.randint(1, 10 ** 1)
        l = self.random.randint(1, 10)
        self.add(dict(a=a, r=r, l=l))


class LineIntersection(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(e: List[int], a=2, b=-1, c=1, d=2021):
        """
        Find the intersection of two lines.
        Solution should be a list of the (x,y) coordinates.
        Accuracy of fifth decimal digit is required.
        """
        x = e[0] / e[1]
        return abs(a * x + b - c * x - d) < 10 ** -5

    @staticmethod
    def sol(a, b, c, d):
        return [d - b, a - c]

    def gen_random(self):
        a = self.random.randint(-10 ** 8, 10 ** 8)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        c = a
        while c == a:
            c = self.random.randint(-10 ** 8, 10 ** 8)
        d = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b, c=c, d=d))


class IfProblem(PuzzleGenerator):

    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, a=324554, b=1345345):
        """Satisfy a simple if statement"""
        if a < 50:
            return x + a == b
        else:
            return x - 2 * a == b

    @staticmethod
    def sol(a, b):
        if a < 50:
            return b - a
        else:
            return b + 2 * a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfProblemWithAnd(PuzzleGenerator):

    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, a=9384594, b=1343663):
        """Satisfy a simple if statement with an and clause"""
        if x > 0 and a > 50:
            return x - a == b
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a > 50 and b > a:
            return b + a
        else:
            return b - a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(0, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfProblemWithOr(PuzzleGenerator):

    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, a=253532, b=1230200):
        """Satisfy a simple if statement with an or clause"""
        if x > 0 or a > 50:
            return x - a == b
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a > 50 or b > a:
            return b + a
        else:
            return b - a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfCases(PuzzleGenerator):

    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, a=4, b=54368639):
        """Satisfy a simple if statement with multiple cases"""
        if a == 1:
            return x % 2 == 0
        elif a == -1:
            return x % 2 == 1
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a == 1:
            x = 0
        elif a == -1:
            x = 1
        else:
            x = b - a
        return x

    def gen_random(self):
        a = self.random.randint(-5, 5)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class ListPosSum(PuzzleGenerator):

    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], n=5, s=19):
        """Find a list of n non-negative integers that sum up to s"""
        return len(x) == n and sum(x) == s and all([a > 0 for a in x])

    @staticmethod
    def sol(n, s):
        x = [1] * n
        x[0] = s - n + 1
        return x

    def gen_random(self):
        n = self.random.randint(1, 10 ** 2)
        s = self.random.randint(n, 10 ** 8)
        self.add(dict(n=n, s=s))


class ListDistinctSum(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], n=4, s=2021):
        """Construct a list of n distinct integers that sum up to s"""
        return len(x) == n and sum(x) == s and len(set(x)) == n

    @staticmethod
    def sol(n, s):
        a = 1
        x = []
        while len(x) < n - 1:
            x.append(a)
            a = -a
            if a in x:
                a += 1

        if s - sum(x) in x:
            x = [i for i in range(n - 1)]

        x = x + [s - sum(x)]
        return x

    def gen_random(self):
        n = self.random.randint(1, 10 ** 3)
        s = self.random.randint(n + 1, 10 ** 8)
        self.add(dict(n=n, s=s))


class ConcatStrings(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(x: str, s=["a", "b", "c", "d", "e", "f"], n=4):
        """Concatenate the list of characters in s"""
        return len(x) == n and all([x[i] == s[i] for i in range(n)])

    @staticmethod
    def sol(s, n):
        return ''.join([s[i] for i in range(n)])

    def gen_random(self):
        n = self.random.randint(0, 25)
        extra = self.random.randint(0, 25)
        s = [self.random.char() for _ in range(n + extra)]
        self.add(dict(n=n, s=s))


class SublistSum(PuzzleGenerator):

    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], t=100, a=25, e=125, s=25):
        """Sum values of sublist by range specifications"""
        non_zero = [z for z in x if z != 0]
        return t == sum([x[i] for i in range(a, e, s)]) and len(set(non_zero)) == len(non_zero) and all(
            [x[i] != 0 for i in range(a, e, s)])

    @staticmethod
    def sol(t, a, e, s):
        x = [0] * e
        for i in range(a, e, s):
            x[i] = i
        correction = t - sum(x) + x[i]

        if correction in x:
            x[correction] = -1 * correction
            x[i] = 3 * correction
        else:
            x[i] = correction
        return x

    def gen_random(self):
        t = self.random.randint(1, 500)
        a = self.random.randint(1, 100)
        e = self.random.randint(a+1, 10 ** 2)
        s = self.random.randint(1, 10)
        self.add(dict(t=t, a=a, e=e, s=s))


class CumulativeSum(PuzzleGenerator):

    tags = [Tags.math, Tags.trivial]

    @staticmethod
    def sat(x: List[int], t=50, n=10):
        """Find how many values have cumulative sum less than target"""
        assert all([v > 0 for v in x])
        s = 0
        i = 0
        for v in sorted(x):
            s += v
            if s > t:
                return i == n
            i += 1
        return i == n

    @staticmethod
    def sol(t, n):
        return [1]*n + [t]

    def gen_random(self):
        n = self.random.randint(1, 10)
        t = self.random.randint(n, 10 ** 2)
        self.add(dict(t=t, n=n))


class BasicStrCounts(PuzzleGenerator):

    tags = [Tags.strings]

    @staticmethod
    def sat(s: str, s1='a', s2='b', count1=50, count2=30):
        """
        Find a string that has count1 occurrences of s1 and count2 occurrences of s2 and starts and ends with
        the same 10 characters
        """
        return s.count(s1) == count1 and s.count(s2) == count2 and s[:10] == s[-10:]

    @staticmethod
    def sol(s1, s2, count1, count2):
        if s1 == s2:
            ans = (s1 + "?") * count1
        elif s1.count(s2):
            ans = (s1 + "?") * count1
            ans += (s2 + "?") * (count2 - ans.count(s2))
        else:
            ans = (s2 + "?") * count2
            ans += (s1 + "?") * (count1 - ans.count(s1))
        return "?" * 10 + ans + "?" * 10

    def gen_random(self):
        s1 = self.random.pseudo_word(max_len=3)
        s2 = self.random.pseudo_word(max_len=3)
        count1 = self.random.randrange(100)
        count2 = self.random.randrange(100)
        inputs = dict(s1=s1, s2=s2, count1=count1, count2=count2)
        if self.sat(self.sol(**inputs), **inputs):
            self.add(inputs)


class ZipStr(PuzzleGenerator):

    tags = [Tags.strings, Tags.trivial]

    @staticmethod
    def sat(s: str, substrings=["foo", "bar", "baz", "oddball"]):
        """
        Find a string that contains each string in substrings alternating, e.g., 'cdaotg' for 'cat' and 'dog'
        """
        return all(sub in s[i::len(substrings)] for i, sub in enumerate(substrings))

    @staticmethod
    def sol(substrings):
        m = max(len(s) for s in substrings)
        return "".join([(s[i] if i < len(s) else " ") for i in range(m) for s in substrings])

    def gen_random(self):
        substrings = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 5))]
        self.add(dict(substrings=substrings))


class ReverseCat(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(s: str, substrings=["foo", "bar", "baz"]):
        """
        Find a string that contains all the substrings reversed and forward
        """
        return all(sub in s and sub[::-1] in s for sub in substrings)

    @staticmethod
    def sol(substrings):
        return "".join(substrings + [s[::-1] for s in substrings])

    def gen_random(self):
        substrings = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 5))]
        self.add(dict(substrings=substrings))


class EngineerNumbers(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(ls: List[str], n=10, a='bar', b='foo'):
        """
        Find a list of n strings, in alphabetical order, starting with a and ending with b.
        """
        return len(ls) == len(set(ls)) == n and ls[0] == a and ls[-1] == b and ls == sorted(ls)

    @staticmethod
    def sol(n, a, b):
        return sorted([a] + [a + chr(0) + str(i) for i in range(n - 2)] + [b])

    def gen_random(self):
        a, b = sorted(self.random.pseudo_word() for _ in range(2))
        n = self.random.randrange(2, 10)
        if a != b:
            self.add(dict(n=n, a=a, b=b))


class PenultimateString(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(s: str, strings=["cat", "dog", "bird", "fly", "moose"]):
        """Find the alphabetically second to last last string in a list."""
        return s in strings and sum(t > s for t in strings) == 1

    @staticmethod
    def sol(strings):
        return sorted(strings)[-2]

    def gen_random(self):
        strings = [self.random.pseudo_word() for _ in range(10)]
        if self.sat(self.sol(strings), strings=strings):
            self.add(dict(strings=strings))


class PenultimateRevString(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(s: str, strings=["cat", "dog", "bird", "fly", "moose"]):
        """Find the reversed version of the alphabetically second string in a list."""
        return s[::-1] in strings and sum(t < s[::-1] for t in strings) == 1

    @staticmethod
    def sol(strings):
        return sorted(strings)[1][::-1]

    def gen_random(self):
        strings = [self.random.pseudo_word() for _ in range(10)]
        if self.sat(self.sol(strings), strings=strings):
            self.add(dict(strings=strings))


class CenteredString(PuzzleGenerator):

    tags = [Tags.trivial, Tags.strings]

    @staticmethod
    def sat(s: str, target="foobarbazwow", length=6):
        """Find a substring of the given length centered within the target string."""
        return target[(len(target) - length) // 2:(len(target) + length) // 2] == s

    @staticmethod
    def sol(target, length):
        return target[(len(target) - length) // 2:(len(target) + length) // 2]

    def gen_random(self):
        target = self.random.pseudo_word()
        length = self.random.randrange(len(target), 0, -1)
        self.add(dict(target=target, length=length))


class SubstrCount(PuzzleGenerator):

    tags = [Tags.brute_force, Tags.strings]

    @staticmethod
    def sat(substring: str, string="moooboooofasd", count=2):
        """Find a substring with a certain count in a given string"""
        return string.count(substring) == count

    @staticmethod
    def sol(string, count):
        for i in range(len(string)):
            for j in range(i+1, len(string)):
                substring = string[i:j]
                c = string.count(substring)
                if c == count:
                    return substring
                if c < count:
                    break
        assert False

    def gen_random(self):
        string = self.random.pseudo_word(max_len=self.random.randrange(1, 100))
        candidates = [string[i:j] for i in range(len(string)) for j in range(i, len(string)) if
                      len(string[i:j]) > 1 and string.count(string[i:j]) > 2]
        if not candidates:
            return
        substring = self.random.choice(candidates)
        count = string.count(substring)
        self.add(dict(string=string, count=count))


class CompleteParens(PuzzleGenerator):

    tags = []

    @staticmethod
    def sat(t: str, s="))(Add)some))parens()to()(balance(()(()(me!)(((("):
        """Add parentheses to the beginning and end of s to make all parentheses balanced"""
        for i in range(len(t) + 1):
            depth = t[:i].count("(") - t[:i].count(")")
            assert depth >= 0
        return depth == 0 and s in t

    @staticmethod
    def sol(s):
        return "(" * s.count(")") + s + ")" * s.count("(")

    def gen_random(self):
        t = ""
        depth = 0
        while depth > 0 or self.random.randrange(10):
            t += self.random.choice([self.random.pseudo_word(min_len=0, max_len=3), "(", "("] + [")", ")", ")"] * (depth > 0))
            depth = t.count("(") - t.count(")")
        a, b = sorted([self.random.randrange(len(t) + 1) for _ in range(2)])
        s = t[a:b]
        if 5 < len(s) < 60:
            self.add(dict(s=s))


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
