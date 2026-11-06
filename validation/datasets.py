"""Hand-written in-repo fixtures (no downloads): 20 two-hop QA items and 20 constraint puzzles.

QA entities are fictional so a model without memory cannot answer from pretraining.
Puzzles have a unique integer solution (verified against Z3 by the test-suite).
"""

# ruff: noqa: E501  (puzzle statements are single-line prose)
import json
import random
import re
from dataclasses import dataclass

from chrysalis_mcp.core.task_features import CHARS_PER_TOKEN


@dataclass(frozen=True)
class QaItem:
    triples: tuple[tuple[str, str, str], tuple[str, str, str]]
    question: str
    answer: str


# (subject, rel1, middle, rel2, answer, question)
_QA_ROWS = [
    (
        "Ilmar Voss",
        "works_at",
        "Brindle Optics",
        "headquartered_in",
        "Tallin Reach",
        "In which city is the employer of Ilmar Voss headquartered?",
    ),
    (
        "Sera Quill",
        "founded",
        "Mossgate Press",
        "publishes",
        "tide almanacs",
        "What does the company founded by Sera Quill publish?",
    ),
    (
        "Dorn Achebe",
        "captains",
        "the Kestrel Lantern",
        "sails_from",
        "Port Averin",
        "From which port does the ship captained by Dorn Achebe sail?",
    ),
    (
        "Yula Penrose",
        "mentors",
        "Teo Larkspur",
        "plays",
        "the glass harmonica",
        "Which instrument does the person mentored by Yula Penrose play?",
    ),
    (
        "Hesper Blanc",
        "owns",
        "Cinder Hollow Farm",
        "grows",
        "saffron pears",
        "What crop is grown on the farm owned by Hesper Blanc?",
    ),
    (
        "Marek Oduya",
        "leads",
        "Project Halcyon",
        "funded_by",
        "the Orrin Trust",
        "Who funds the project led by Marek Oduya?",
    ),
    (
        "Lisbet Aro",
        "lives_in",
        "Quenby",
        "lies_on",
        "the Varn river",
        "On which river does the town where Lisbet Aro lives lie?",
    ),
    (
        "Tomas Reyes-Wu",
        "teaches_at",
        "Calloway Institute",
        "founded_in",
        "1874",
        "In what year was the institute where Tomas Reyes-Wu teaches founded?",
    ),
    (
        "Nima Falk",
        "designed",
        "the Lumen Bridge",
        "spans",
        "the Sorrel Gorge",
        "What does the bridge designed by Nima Falk span?",
    ),
    (
        "Ottoline Briggs",
        "directs",
        "Harbor of Echoes",
        "premiered_at",
        "the Dunmere Festival",
        "At which festival did the film directed by Ottoline Briggs premiere?",
    ),
    (
        "Kasimir Ndiaye",
        "pilots",
        "the Zephyr Wren",
        "is_a",
        "autogyro",
        "What type of aircraft does Kasimir Ndiaye pilot?",
    ),
    (
        "Anwen Costa",
        "brews",
        "Amber Fathom",
        "contains",
        "smoked juniper",
        "Which ingredient is in the beer brewed by Anwen Costa?",
    ),
    (
        "Rurik Sandoval",
        "coaches",
        "the Pale Herons",
        "compete_in",
        "the Ninth League",
        "In which league does the team coached by Rurik Sandoval compete?",
    ),
    (
        "Edda Moreau",
        "curates",
        "the Tessel Gallery",
        "located_on",
        "Fennel Street",
        "On which street is the gallery curated by Edda Moreau?",
    ),
    (
        "Jovan Itami",
        "wrote",
        "The Salt Cartographer",
        "set_in",
        "Old Marrow",
        "Where is the novel written by Jovan Itami set?",
    ),
    (
        "Pia Okonkwo-Lund",
        "manages",
        "Greywater Station",
        "monitors",
        "glacier melt",
        "What does the station managed by Pia Okonkwo-Lund monitor?",
    ),
    (
        "Cassian Bell",
        "composed",
        "Vesper in Rust",
        "scored_for",
        "twelve cellos",
        "For which ensemble was the piece composed by Cassian Bell scored?",
    ),
    (
        "Mirela Tran",
        "chairs",
        "the Alder Council",
        "meets_in",
        "the Hollin Chamber",
        "Where does the council chaired by Mirela Tran meet?",
    ),
    (
        "Bruno Esteve",
        "repairs",
        "the Ostrander Clock",
        "stands_in",
        "Linden Square",
        "In which square stands the clock repaired by Bruno Esteve?",
    ),
    (
        "Zofia Kimathi",
        "discovered",
        "comet Vael-3",
        "named_after",
        "a harbor lighthouse",
        "After what is the comet discovered by Zofia Kimathi named?",
    ),
]

QA_ITEMS: tuple[QaItem, ...] = tuple(
    QaItem(((s, r1, mid), (mid, r2, ans)), q, ans) for s, r1, mid, r2, ans, q in _QA_ROWS
)


@dataclass(frozen=True)
class Puzzle:
    text: str
    smt2: str  # reference encoding (declare-const + assert); a unique integer solution
    answer: dict[str, int]


def _p(text: str, decls: str, asserts: list[str], answer: dict[str, int]) -> Puzzle:
    body = "".join(f"(declare-const {v} Int)" for v in decls.split()) + "".join(
        f"(assert {a})" for a in asserts
    )
    return Puzzle(text, body, answer)


_BOX = "(and (>= {v} 1) (<= {v} 9))"
PUZZLES: tuple[Puzzle, ...] = (
    _p(
        "Two integers sum to 10 and differ by 2. Find x > y.",
        "x y",
        ["(= (+ x y) 10)", "(= (- x y) 2)"],
        {"x": 6, "y": 4},
    ),
    _p(
        "x*y = 35, x + y = 12, x > y, both positive.",
        "x y",
        ["(= (* x y) 35)", "(= (+ x y) 12)", "(> x y)", "(> y 0)"],
        {"x": 7, "y": 5},
    ),
    _p(
        "Three integers in 1..9 are all different, sum to 24, and x > y > z.",
        "x y z",
        [_BOX.format(v=v) for v in "xyz"] + ["(= (+ x y z) 24)", "(> x y)", "(> y z)"],
        {"x": 9, "y": 8, "z": 7},
    ),
    _p(
        "x is twice y, and y plus 7 equals x.",
        "x y",
        ["(= x (* 2 y))", "(= (+ y 7) x)"],
        {"x": 14, "y": 7},
    ),
    _p(
        "A two-digit number ab (a tens, b units; a in 1..9, b in 0..9) has digit sum 11 and "
        "ab - ba = 27. Give a and b.",
        "a b",
        [
            "(and (>= a 1) (<= a 9))",
            "(and (>= b 0) (<= b 9))",
            "(= (+ a b) 11)",
            "(= (- (+ (* 10 a) b) (+ (* 10 b) a)) 27)",
        ],
        {"a": 7, "b": 4},
    ),
    _p(
        "x mod 7 = 3, x mod 5 = 2, 0 < x < 35.",
        "x",
        ["(= (mod x 7) 3)", "(= (mod x 5) 2)", "(> x 0)", "(< x 35)"],
        {"x": 17},
    ),
    _p(
        "Pens cost 3 and notebooks 5; 4 items total cost 14. Give p (pens) and n (notebooks).",
        "p n",
        ["(= (+ p n) 4)", "(= (+ (* 3 p) (* 5 n)) 14)", "(>= p 0)", "(>= n 0)"],
        {"p": 3, "n": 1},
    ),
    _p(
        "x^2 + y^2 = 25, x > y > 0.",
        "x y",
        ["(= (+ (* x x) (* y y)) 25)", "(> x y)", "(> y 0)"],
        {"x": 4, "y": 3},
    ),
    _p(
        "Ana is 6 years older than Ben; in 4 years the sum of their ages is 40. Give ana, ben "
        "(current ages).",
        "ana ben",
        ["(= ana (+ ben 6))", "(= (+ (+ ana 4) (+ ben 4)) 40)"],
        {"ana": 19, "ben": 13},
    ),
    _p(
        "x + y + z = 6 and x*y*z = 6 with 1 <= x < y < z.",
        "x y z",
        ["(= (+ x y z) 6)", "(= (* x y z) 6)", "(>= x 1)", "(< x y)", "(< y z)"],
        {"x": 1, "y": 2, "z": 3},
    ),
    _p(
        "A rectangle has perimeter 22 and area 30; w < l, positive integers.",
        "w l",
        ["(= (* 2 (+ w l)) 22)", "(= (* w l) 30)", "(< w l)", "(> w 0)"],
        {"w": 5, "l": 6},
    ),
    _p(
        "x is a multiple of 6, 40 < x < 50, and x - 2 is divisible by 5.",
        "x",
        ["(= (mod x 6) 0)", "(> x 40)", "(< x 50)", "(= (mod (- x 2) 5) 0)"],
        {"x": 42},
    ),
    _p(
        "Three consecutive integers sum to 48; give the smallest, s.",
        "s",
        ["(= (+ s (+ s 1) (+ s 2)) 48)"],
        {"s": 15},
    ),
    _p(
        "x - 2y = 1 and 3x + y = 17.",
        "x y",
        ["(= (- x (* 2 y)) 1)", "(= (+ (* 3 x) y) 17)"],
        {"x": 5, "y": 2},
    ),
    _p(
        "A coin jar holds only 2s and 5s: 9 coins worth 33. Give t (twos) and f (fives).",
        "t f",
        ["(= (+ t f) 9)", "(= (+ (* 2 t) (* 5 f)) 33)", "(>= t 0)", "(>= f 0)"],
        {"t": 4, "f": 5},
    ),
    _p(
        "x > 0, x*x - 9*x + 14 = 0 and x > 5.",
        "x",
        ["(> x 5)", "(= (+ (- (* x x) (* 9 x)) 14) 0)"],
        {"x": 7},
    ),
    _p(
        "a + b = 11, b + c = 13, a + c = 10.",
        "a b c",
        ["(= (+ a b) 11)", "(= (+ b c) 13)", "(= (+ a c) 10)"],
        {"a": 4, "b": 7, "c": 6},
    ),
    _p(
        "A number n in 1..99 is a perfect square, odd, and its digits sum to 9, n > 50.",
        "n",
        [
            "(and (>= n 51) (<= n 99))",
            "(= (mod n 2) 1)",
            "(= (+ (div n 10) (mod n 10)) 9)",
            "(exists ((k Int)) (= n (* k k)))",
        ],
        {"n": 81},
    ),
    _p(
        "x in 0..20, x is even, x mod 3 = 1 and x mod 4 = 2.",
        "x",
        ["(and (>= x 0) (<= x 20))", "(= (mod x 2) 0)", "(= (mod x 3) 1)", "(= (mod x 4) 2)"],
        {"x": 10},
    ),
    _p("x*(x+1) = 56 and x > 0.", "x", ["(= (* x (+ x 1)) 56)", "(> x 0)"], {"x": 7}),
)


@dataclass(frozen=True)
class HardPuzzle:
    """A bounded integer puzzle for the reason-tool DSL; ``answer`` is the unique optimum/solution.

    ``assert_`` and ``goal`` are the reference encoding in the ``constraints`` JSON DSL (vars are
    all ``int``); the test-suite proves with Z3 that exactly one assignment satisfies it.
    """

    text: str
    assert_: tuple[str, ...]
    goal: str | None
    answer: dict[str, int]

    @property
    def constraints(self) -> dict:
        spec: dict = {"vars": {v: "int" for v in self.answer}, "assert": list(self.assert_)}
        if self.goal:
            spec["goal"] = self.goal
        return spec


def _h(text: str, asserts: list[str], answer: dict[str, int], goal: str | None = None):
    return HardPuzzle(text, tuple(asserts), goal, answer)


def _rng(v: str, lo: int, hi: int) -> list[str]:
    return [f"{v} >= {lo}", f"{v} <= {hi}"]


def _rngs(vs: str, lo: int, hi: int) -> list[str]:
    return [c for v in vs for c in _rng(v, lo, hi)]


def _distinct(vs: str) -> list[str]:
    return [f"{a} != {b}" for i, a in enumerate(vs) for b in vs[i + 1 :]]


HARD_PUZZLES: tuple[HardPuzzle, ...] = (
    _h(
        "Integers a, b, c, each between 0 and 20: a + b + c = 28, a is twice b, c is b plus 4.",
        [*_rngs("abc", 0, 20), "a + b + c == 28", "a == 2*b", "c == b + 4"],
        {"a": 12, "b": 6, "c": 10},
    ),
    _h(
        "Integers p, q between 1 and 30: p*q = 72, 18 <= p + q <= 22, p < q, and q - p <= 6.",
        [*_rngs("pq", 1, 30), "p*q == 72", "p + q <= 22", "p + q >= 18", "p < q", "q - p <= 6"],
        {"p": 6, "q": 12},
    ),
    _h(
        "Integers a, b, c, d between 1 and 9, all different. a + b = c + d, a*b = 12, c*d = 10, "
        "a < b and c > d.",
        [
            *_rngs("abcd", 1, 9),
            *_distinct("abcd"),
            "a + b == c + d",
            "a*b == 12",
            "c > d",
            "a < b",
            "c*d == 10",
        ],
        {"a": 3, "b": 4, "c": 5, "d": 2},
    ),
    _h(
        "Integers x, y between 0 and 20 with 3x + 2y <= 30, x + 4y <= 36 and x + y >= 5. "
        "Among all solutions take the one with the largest x.",
        [*_rngs("xy", 0, 20), "3*x + 2*y <= 30", "x + 4*y <= 36", "x + y >= 5"],
        {"x": 10, "y": 0},
        "max:x",
    ),
    _h(
        "Integers x, y, z between 0 and 10: x + y + z = 12, x*y = z + 9, x <= y.",
        [*_rngs("xyz", 0, 10), "x + y + z == 12", "x*y == z + 9", "x <= y"],
        {"x": 1, "y": 10, "z": 1},
    ),
    _h(
        "Integers x, y between 0 and 25: 2x + 3y = 41 and x > y. Take the solution with the "
        "smallest y.",
        [*_rngs("xy", 0, 25), "2*x + 3*y == 41", "x > y"],
        {"x": 19, "y": 1},
        "min:y",
    ),
    _h(
        "Integers a, b, c between 0 and 15 with a < b < c, a + b + c = 24 and a*c = b*b - 1.",
        [*_rngs("abc", 0, 15), "a < b", "b < c", "a + b + c == 24", "a*c == b*b - 1"],
        {"a": 7, "b": 8, "c": 9},
    ),
    _h(
        "You buy p pens at 2, n pads at 3 and k cases at 7 (1 to 8 of each), spending exactly "
        "40. Among all ways take the one with the most pens.",
        [*_rngs("pnk", 1, 8), "2*p + 3*n + 7*k == 40"],
        {"p": 8, "n": 1, "k": 3},
        "max:p",
    ),
    _h(
        "Integers x, y, z between 1 and 12: x + y = z, x*y = 2z + 6, x <= y.",
        [*_rngs("xyz", 1, 12), "x + y == z", "x*y == 2*z + 6", "x <= y"],
        {"x": 4, "y": 7, "z": 11},
    ),
    _h(
        "Integers a, b, c, d between 0 and 9: a + b = 10, c + d = 7, a*c = 15, b - d = 5.",
        [*_rngs("abcd", 0, 9), "a + b == 10", "c + d == 7", "a*c == 15", "b - d == 5"],
        {"a": 3, "b": 7, "c": 5, "d": 2},
    ),
    _h(
        "Integers x, y, z between 0 and 20: x + y + z = 20, x >= y + 2, y >= z + 2, z >= 1. "
        "Take the solution with the smallest x.",
        [*_rngs("xyz", 0, 20), "x + y + z == 20", "x >= y + 2", "y >= z + 2", "z >= 1"],
        {"x": 9, "y": 7, "z": 4},
        "min:x",
    ),
    _h(
        "Integers m, n between 1 and 40: m*n = 120, m + n = 26, m < n.",
        [*_rngs("mn", 1, 40), "m*n == 120", "m + n == 26", "m < n"],
        {"m": 6, "n": 20},
    ),
    _h(
        "Integers a, b, c between 1 and 10: either a = b + c or a*b = c (or both), "
        "a + b + c = 12, 2 <= b < c and a > c.",
        [
            *_rngs("abc", 1, 10),
            "a == b + c or a*b == c",
            "a + b + c == 12",
            "b >= 2",
            "b < c",
            "a > c",
        ],
        {"a": 6, "b": 2, "c": 4},
    ),
    _h(
        "Integers x, y between 0 and 30: x*y = 36, x > y, and x - y is not 5. Take the solution "
        "with the largest x.",
        [*_rngs("xy", 0, 30), "x*y == 36", "x > y", "not (x - y == 5)"],
        {"x": 18, "y": 2},
        "max:x",
    ),
    _h(
        "Integers u, v, w between 1 and 9 with u < v < w, u + v + w = 16 and u*v*w = 120.",
        [*_rngs("uvw", 1, 9), "u < v", "v < w", "u + v + w == 16", "u*v*w == 120"],
        {"u": 3, "v": 5, "w": 8},
    ),
    _h(
        "a, b, c, d, e form a permutation of 1..5 with a < b, c > d, e = 3 and a + c = 7.",
        [
            *_rngs("abcde", 1, 5),
            *_distinct("abcde"),
            "a < b",
            "c > d",
            "e == 3",
            "a + c == 7",
        ],
        {"a": 2, "b": 4, "c": 5, "d": 1, "e": 3},
    ),
    _h(
        "Integers x, y between 0 and 50 with x + y <= 30, 2x + y >= 30 and y >= 4. Take the "
        "solution with the largest x.",
        [*_rngs("xy", 0, 50), "x + y <= 30", "2*x + y >= 30", "y >= 4"],
        {"x": 26, "y": 4},
        "max:x",
    ),
)

JSON_OBJ = re.compile(r"\{[^{}]*\}")


# Held-out set for the T13 circularity check: generated once with random.Random(777) (rejection
# sampling over a brute-forced search space, unique solution/optimum), never used to fit the
# advisory thresholds in data/evidence.json. n_atoms is measured with the same regex as the
# advisor; the test-suite re-proves uniqueness with Z3 exactly as for HARD_PUZZLES.
HELDOUT_HARD: tuple[HardPuzzle, ...] = (
    _h(
        "Integers d between 2 and 8, p between 2 and 8, y between 2 and 6. Constraints: y + p <= 13; d + p + y == 18; p == 7; y + d <= 11; y - p == -1.",
        [
            "d >= 2",
            "d <= 8",
            "p >= 2",
            "p <= 8",
            "y >= 2",
            "y <= 6",
            "y + p <= 13",
            "d + p + y == 18",
            "p == 7",
            "y + d <= 11",
            "y - p == -1",
        ],
        {"d": 5, "p": 7, "y": 6},
    ),
    _h(
        "Integers q between 1 and 5, z between 1 and 6, c between 1 and 7, a between 1 and 5. Constraints: c*q == 20; 2*a + c == 15; c + a + z == 15.",
        [
            "q >= 1",
            "q <= 5",
            "z >= 1",
            "z <= 6",
            "c >= 1",
            "c <= 7",
            "a >= 1",
            "a <= 5",
            "c*q == 20",
            "2*a + c == 15",
            "c + a + z == 15",
        ],
        {"q": 4, "z": 5, "c": 5, "a": 5},
    ),
    _h(
        "Integers x between 1 and 6, y between 1 and 5, a between 1 and 6. Constraints: 2*x + y == 16; a + x <= 9; y < x; x > a; a < y; x + a >= 9. Choose the solution with the smallest y.",
        [
            "x >= 1",
            "x <= 6",
            "y >= 1",
            "y <= 5",
            "a >= 1",
            "a <= 6",
            "2*x + y == 16",
            "a + x <= 9",
            "y < x",
            "x > a",
            "a < y",
            "x + a >= 9",
        ],
        {"x": 6, "y": 4, "a": 3},
        "min:y",
    ),
    _h(
        "Integers c between 2 and 6, f between 2 and 9, g between 2 and 9, x between 2 and 6. Constraints: f + g + x == 13; c - x == -1; x + g + c == 11. Choose the solution with the largest x.",
        [
            "c >= 2",
            "c <= 6",
            "f >= 2",
            "f <= 9",
            "g >= 2",
            "g <= 9",
            "x >= 2",
            "x <= 6",
            "f + g + x == 13",
            "c - x == -1",
            "x + g + c == 11",
        ],
        {"c": 4, "f": 6, "g": 2, "x": 5},
        "max:x",
    ),
    _h(
        "Integers x between 1 and 5, c between 1 and 8, y between 1 and 6, f between 1 and 5. Constraints: f + x + y == 5; f - x == 1; 2*x + c == 6.",
        [
            "x >= 1",
            "x <= 5",
            "c >= 1",
            "c <= 8",
            "y >= 1",
            "y <= 6",
            "f >= 1",
            "f <= 5",
            "f + x + y == 5",
            "f - x == 1",
            "2*x + c == 6",
        ],
        {"x": 1, "c": 4, "y": 2, "f": 2},
    ),
    _h(
        "Integers r between 3 and 7, q between 3 and 10, e between 3 and 7. Constraints: r == e; r + q == 14; 2*q + e == 22.",
        [
            "r >= 3",
            "r <= 7",
            "q >= 3",
            "q <= 10",
            "e >= 3",
            "e <= 7",
            "r == e",
            "r + q == 14",
            "2*q + e == 22",
        ],
        {"r": 6, "q": 8, "e": 6},
    ),
    _h(
        "Integers z between 3 and 8, c between 3 and 8, q between 3 and 7, r between 3 and 8. Constraints: c + z == 13; c + q >= 12; q + c + r == 15. Choose the solution with the smallest c.",
        [
            "z >= 3",
            "z <= 8",
            "c >= 3",
            "c <= 8",
            "q >= 3",
            "q <= 7",
            "r >= 3",
            "r <= 8",
            "c + z == 13",
            "c + q >= 12",
            "q + c + r == 15",
        ],
        {"z": 8, "c": 5, "q": 7, "r": 3},
        "min:c",
    ),
    _h(
        "Integers g between 1 and 6, e between 1 and 8, y between 1 and 5. Constraints: 2*e + y == 16; y == 4; y + e + g == 12. Choose the solution with the smallest g.",
        [
            "g >= 1",
            "g <= 6",
            "e >= 1",
            "e <= 8",
            "y >= 1",
            "y <= 5",
            "2*e + y == 16",
            "y == 4",
            "y + e + g == 12",
        ],
        {"g": 2, "e": 6, "y": 4},
        "min:g",
    ),
)
HELDOUT_EASY: tuple[HardPuzzle, ...] = (
    _h(
        "Integers r between 2 and 8, x between 2 and 7. Constraints: 2*x + r == 18; r + x == 13.",
        ["r >= 2", "r <= 8", "x >= 2", "x <= 7", "2*x + r == 18", "r + x == 13"],
        {"r": 8, "x": 5},
    ),
    _h(
        "Integers d between 3 and 9, e between 3 and 10. Constraints: 2*e + d == 15; d*e == 27.",
        ["d >= 3", "d <= 9", "e >= 3", "e <= 10", "2*e + d == 15", "d*e == 27"],
        {"d": 9, "e": 3},
    ),
    _h(
        "Integers e between 2 and 6, c between 2 and 9. Constraints: 2*c + e == 7; e + c == 5.",
        ["e >= 2", "e <= 6", "c >= 2", "c <= 9", "2*c + e == 7", "e + c == 5"],
        {"e": 3, "c": 2},
    ),
    _h(
        "Integers q between 3 and 10, y between 3 and 10. Constraints: q == 8; 2*q + y == 24.",
        ["q >= 3", "q <= 10", "y >= 3", "y <= 10", "q == 8", "2*q + y == 24"],
        {"q": 8, "y": 8},
    ),
    _h(
        "Integers c between 1 and 8, g between 1 and 6. Constraints: c + g == 2.",
        ["c >= 1", "c <= 8", "g >= 1", "g <= 6", "c + g == 2"],
        {"c": 1, "g": 1},
    ),
    _h(
        "Integers z between 1 and 7, d between 1 and 6. Constraints: z - d == 0; 2*z + d == 6.",
        ["z >= 1", "z <= 7", "d >= 1", "d <= 6", "z - d == 0", "2*z + d == 6"],
        {"z": 2, "d": 2},
    ),
    _h(
        "Integers q between 3 and 8, a between 3 and 9 (plus one more integer fixed below). Constraints: y == 7; q + a <= 6.",
        ["q >= 3", "q <= 8", "a >= 3", "a <= 9", "y == 7", "q + a <= 6"],
        {"q": 3, "a": 3, "y": 7},
    ),
    _h(
        "Integers b between 3 and 10, f between 3 and 7 (plus one more integer fixed below). Constraints: c == 9; c + f + b == 15.",
        ["b >= 3", "b <= 10", "f >= 3", "f <= 7", "c == 9", "c + f + b == 15"],
        {"b": 3, "f": 3, "c": 9},
    ),
)
HELDOUT_PUZZLES: tuple[HardPuzzle, ...] = (*HELDOUT_HARD, *HELDOUT_EASY)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", text.casefold()).split())


def score_qa(answer: str | None, gold: str) -> bool:
    """Exact match after case/punctuation folding, tolerating a leading article."""
    if not answer:
        return False
    got = re.sub(r"^(the|a|an)\s+", "", normalize(answer))
    return got == re.sub(r"^(the|a|an)\s+", "", normalize(gold))


def score_puzzle(answer: str | None, gold: dict[str, int]) -> bool:
    """The last JSON object in the answer must equal the gold assignment."""
    for raw in reversed(JSON_OBJ.findall(answer or "")):
        try:
            got = json.loads(raw)
        except ValueError:
            continue
        if isinstance(got, dict):
            try:
                return {k: int(v) for k, v in got.items()} == gold
            except TypeError, ValueError:
                return False
    return False


def synthetic_text(rng: random.Random, vocab: list[str], n_words: int = 6) -> str:
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def synthetic_vocab(rng: random.Random, size: int = 400) -> list[str]:
    syll = ["ka", "to", "ri", "mel", "von", "sa", "du", "lex", "pi", "no", "ar", "zel", "qui", "bo"]
    return sorted(
        {"".join(rng.choice(syll) for _ in range(rng.randint(2, 3))) for _ in range(size)}
    )


# --- H-a-scale: two-hop QA with more facts than the model's context window ----------------------

SCALE_CHAINS = 400  # 2 facts each: ~800 facts, about 1.7x an 8192-token window
SCALE_QUESTIONS = 20
SCALE_SEED = 13
_NAME_SYLLABLES = (
    "bri",
    "ndle",
    "mos",
    "gate",
    "ker",
    "tel",
    "ave",
    "rin",
    "lan",
    "yul",
    "pen",
    "ros",
)
_NAME_SYLLABLES += (
    "tao",
    "lar",
    "kes",
    "ach",
    "ebe",
    "dor",
    "han",
    "sel",
    "ovi",
    "quo",
    "var",
    "nim",
)


@dataclass(frozen=True)
class ScaledQa:
    """``triples``/``facts`` are shuffled (the two hops of a chain are far apart); ``items`` are
    the questions drawn from distinct chains."""

    triples: tuple[tuple[str, str, str], ...]
    facts: tuple[str, ...]
    items: tuple[QaItem, ...]


def fact_sentence(s: str, r: str, o: str) -> str:
    return f"{s} {r.replace('_', ' ')} {o}."


def scaled_qa(
    n_chains: int = SCALE_CHAINS, n_questions: int = SCALE_QUESTIONS, seed: int = SCALE_SEED
) -> ScaledQa:
    """Deterministic fictional two-hop chains (same relations/questions as the base set)."""
    if not 0 < n_questions <= n_chains:
        raise ValueError("need 0 < n_questions <= n_chains")
    rng = random.Random(seed)
    used: set[str] = set()

    def name() -> str:
        while True:
            words = [
                "".join(rng.choice(_NAME_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()
                for _ in range(2)
            ]
            full = " ".join(words)
            if full not in used:
                used.add(full)
                return full

    templates = [(r1, r2, q.replace(s, "{s}")) for s, r1, _, r2, _, q in _QA_ROWS]
    chains: list[QaItem] = []
    for i in range(n_chains):
        r1, r2, question = templates[i % len(templates)]
        s, mid, ans = name(), name(), name()
        chains.append(QaItem(((s, r1, mid), (mid, r2, ans)), question.format(s=s), ans))
    triples = [t for c in chains for t in c.triples]
    rng.shuffle(triples)
    asked = rng.sample(range(n_chains), n_questions)
    return ScaledQa(
        triples=tuple(triples),
        facts=tuple(fact_sentence(*t) for t in triples),
        items=tuple(chains[i] for i in sorted(asked)),
    )


def fit_context(
    facts: tuple[str, ...], budget_tokens: int, chars_per_token: float = CHARS_PER_TOKEN
) -> tuple[list[str], float]:
    """Most recent facts (the tail) whose text fits ``budget_tokens``; and the dropped fraction."""
    budget = int(budget_tokens * chars_per_token)
    kept: list[str] = []
    for fact in reversed(facts):
        budget -= len(fact) + 1  # newline
        if budget < 0:
            break
        kept.append(fact)
    kept.reverse()
    return kept, round(1 - len(kept) / len(facts), 4) if facts else 0.0


def answerable_in(items: tuple[QaItem, ...], kept: list[str]) -> float:
    """Share of questions whose two supporting facts both survived truncation."""
    have = set(kept)
    ok = sum(all(fact_sentence(*t) in have for t in item.triples) for item in items)
    return round(ok / len(items), 4) if items else 0.0


_MD_NOISE = re.compile(r"[*$`\\]")
_ASSIGN = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*(?<![=<>!])=(?!=)\s*(-?\d+)(?!\d|\.\d)")


def score_puzzle_values(answer: str | None, gold: dict[str, int]) -> bool:
    """DIAGNOSTIC, format-lenient: strict JSON OR ``name = int`` assignments (Markdown/LaTeX ok).

    First occurrence per gold variable; every gold variable must match exactly. Never used for
    verdicts: it only separates "wrong values" from "right values in the wrong format".
    """
    if score_puzzle(answer, gold):
        return True
    found: dict[str, int] = {}
    for name, value in _ASSIGN.findall(_MD_NOISE.sub("", answer or "")):
        if name in gold:
            found.setdefault(name, int(value))
    return found == gold


# --- H-b end to end (T19): user vocabulary (aliases) that is not in the knowledge base ------------

ALIAS_SEED = 19
N_ALIAS_ENTITIES = 24
N_INTERFERENCE = 300  # extra unrelated ``update`` pairs that stress retention
_ENTITY_STEMS = (
    "kes",
    "tre",
    "vor",
    "mal",
    "dun",
    "ryx",
    "ophi",
    "zan",
    "bel",
    "quor",
    "hesk",
    "ul",
    "ner",
    "tav",
    "iro",
    "sul",
)
_ENTITY_KINDS = ("Station", "Reach", "Hold", "Landing", "Outpost", "Annex", "Cairn", "Depot")
_ALIAS_COLORS = (
    "Amber",
    "Azure",
    "Crimson",
    "Ivory",
    "Jade",
    "Violet",
    "Scarlet",
    "Cobalt",
    "Ochre",
    "Silver",
    "Golden",
    "Copper",
    "Indigo",
    "Teal",
    "Russet",
    "Onyx",
    "Coral",
    "Umber",
    "Pearl",
    "Saffron",
    "Maroon",
    "Lilac",
    "Sable",
    "Ashen",
)
_ALIAS_NOUNS = (
    "Hive",
    "Anvil",
    "Lantern",
    "Orchard",
    "Compass",
    "Loom",
    "Furnace",
    "Cradle",
    "Beacon",
    "Mirror",
    "Thicket",
    "Quill",
    "Kettle",
    "Vault",
    "Garden",
    "Drum",
    "Ladder",
    "Crown",
    "Well",
    "Bell",
    "Tide",
    "Nest",
    "Forge",
    "Wheel",
)
# relation -> (teach template, test template); {a} is the alias, never the entity name
_ALIAS_ASKS: dict[str, tuple[str, str]] = {
    "has_population": ("population of the {a}", "How many people live in the {a}?"),
    "has_elevation_m": (
        "elevation of the {a}",
        "How high above sea level, in meters, is the {a}?",
    ),
    "has_founding_year": ("founding year of the {a}", "In what year was the {a} founded?"),
    "has_dock_count": ("docks at the {a}", "How many docks does the {a} have?"),
}
_ALIAS_RELATIONS = tuple(_ALIAS_ASKS)


@dataclass(frozen=True)
class AliasItem:
    """One invented entity, its KB facts, and the user's alias for it (absent from the KB)."""

    entity: str
    triples: tuple[tuple[str, str, str], ...]
    alias: str
    teach: str  # phrasing used when the user corrects the system
    test: str  # a different question that still contains the alias
    answer: str  # object of the first triple (the one the questions ask about)


@dataclass(frozen=True)
class AliasDataset:
    items: tuple[AliasItem, ...]
    set_a: tuple[AliasItem, ...]  # taught first, then tested
    set_b: tuple[AliasItem, ...]  # taught afterwards (interference)
    interference_triples: tuple[tuple[str, str, str], ...]  # unrelated nodes ingested in the KB
    interference_pairs: tuple[tuple[str, str], ...]  # (query, node text) update pairs

    @property
    def kb_triples(self) -> list[tuple[str, str, str]]:
        return [t for i in self.items for t in i.triples] + list(self.interference_triples)


def alias_dataset(
    seed: int = ALIAS_SEED,
    n_entities: int = N_ALIAS_ENTITIES,
    n_interference: int = N_INTERFERENCE,
) -> AliasDataset:
    """Deterministic: invented entity names, alias = unique colour + unique noun, numeric facts."""
    if not 2 <= n_entities <= min(len(_ALIAS_COLORS), len(_ALIAS_NOUNS)) or n_interference < 0:
        raise ValueError("need 2 <= n_entities <= 24 and n_interference >= 0")
    rng = random.Random(seed)
    colors, nouns = list(_ALIAS_COLORS), list(_ALIAS_NOUNS)
    rng.shuffle(colors)
    rng.shuffle(nouns)
    used: set[str] = set()
    values: set[int] = set()

    def entity_name() -> str:
        while True:
            stem = "".join(rng.choice(_ENTITY_STEMS) for _ in range(2)).capitalize()
            name = f"{stem} {rng.choice(_ENTITY_KINDS)}"
            if name not in used:
                used.add(name)
                return name

    def value(lo: int, hi: int) -> int:
        while (v := rng.randint(lo, hi)) in values:
            pass
        values.add(v)
        return v

    ranges = {
        "has_population": (800, 9500),
        "has_elevation_m": (120, 4800),
        "has_founding_year": (1610, 1990),
        "has_dock_count": (3, 90),
    }
    items = []
    for i in range(n_entities):
        entity, alias = entity_name(), f"{colors[i]} {nouns[i]}"
        rels = rng.sample(_ALIAS_RELATIONS, rng.randint(1, 2))
        triples = tuple((entity, r, str(value(*ranges[r]))) for r in rels)
        teach, test = (t.format(a=alias) for t in _ALIAS_ASKS[rels[0]])
        items.append(AliasItem(entity, triples, alias, teach, test, triples[0][2]))
    order = list(items)
    rng.shuffle(order)
    half = len(order) // 2
    vocab = synthetic_vocab(rng)
    filler = [(synthetic_text(rng, vocab), f"filler node {i}") for i in range(n_interference)]
    return AliasDataset(
        items=tuple(items),
        set_a=tuple(order[:half]),
        set_b=tuple(order[half:]),
        interference_triples=tuple(
            (f"filler anchor {i}", "ref", node) for i, (_, node) in enumerate(filler)
        ),
        interference_pairs=tuple(filler),
    )
