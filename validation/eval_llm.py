"""H-a, H-c and the T12 hypotheses end to end with the live model (marker ``llm``).

Groups (``--hyp``): ``ha`` (alone vs memory), ``ctx`` (whole-dataset-in-prompt baseline),
``hascale`` (more facts than the window: truncated context vs memory), ``hc`` (easy puzzles,
alone vs reason), ``hard`` (HARD_PUZZLES, alone vs tool), ``adv`` (advised arm: the harness
declares the model, the tool description carries the rules), ``cal`` (meta.calibrate loop),
``heldout`` (circularity check: alone, tool and advised on
HELDOUT_PUZZLES, 8 hard + 8 easy never used to fit thresholds), ``hbe2e`` (H-b end to end: the
user's invented aliases taught with ``memory.update`` vs an alias triple vs nothing; the harness
prepares each substrate, interference updates included, then the model answers set-A questions
with the memory tool). Skips cleanly
(``status: UNREACHABLE``) when the OpenAI-compatible server is down. Results merge per
mode and group into ``llm.json`` so arms can be run in separate invocations.

    python -m validation.eval_llm --hyp ha,ctx --thinking off
    python -m validation.eval_llm --hyp hard,adv,cal --thinking both
    python -m validation.eval_llm --hyp hascale --thinking off
    python -m validation.eval_llm --hyp heldout --thinking both
    python -m validation.eval_llm --hyp hbe2e --thinking both
"""

import argparse
import asyncio
import contextlib
import contextvars
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from chrysalis_mcp.core import task_features
from validation import RESULTS_DIR
from validation.bench_hb_e2e import ARMS as HBE2E_BASE_ARMS
from validation.bench_hb_e2e import prepare_ops
from validation.bridge import ChrysalisSession, run_agent
from validation.datasets import (
    HARD_PUZZLES,
    HELDOUT_EASY,
    HELDOUT_HARD,
    PUZZLES,
    QA_ITEMS,
    alias_dataset,
    answerable_in,
    fit_context,
    scaled_qa,
    score_puzzle,
    score_puzzle_values,
    score_qa,
    score_qa_lenient,
)
from validation.llm_client import LlmClient

RESULT_FILE = "llm.json"
DEFAULT_GROUPS = ("ha", "hc")
ALL_GROUPS = ("ha", "ctx", "hc", "hard", "adv", "cal", "hascale", "heldout", "hbe2e")
GROUP_NAMES = {
    "ha": "H-a",
    "ctx": "H-a-ctx",
    "hc": "H-c",
    "hard": "H-c-hard",
    "adv": "H-adv",
    "cal": "calibration",
    "hascale": "H-a-scale",
    "heldout": "H-adv-heldout",
    "hbe2e": "H-b-e2e",
}
# the three verdict arms plus an informational alias baseline free of update-interference
HBE2E_ARMS = (*HBE2E_BASE_ARMS, "alias_ingest_clean")
HA_MODES = ("off",)  # long-prompt / memory groups run with thinking off only (cost)
LONG_GROUPS = ("ha", "ctx", "hascale")
SERVER_CTX_ENV, DEFAULT_SERVER_CTX = "CHRYSALIS_LLM_CTX", 8192
SCALE_RESERVE_TOKENS = 1024  # system prompt + question + answer share the window with the facts
CONTEXT_SEED = 12  # fixed shuffle of the facts in the context arm
DEFAULT_PARAMS_B = 4.0  # Gemma E4B
# (name key, total params B, active params B or None for dense); 26B-A4B is a MoE (class medium)
KNOWN_PARAMS_B: tuple[tuple[str, float, float | None], ...] = (
    ("deepseek-r1-0528-qwen3-8b", 8.0, None),  # before any qwen key: first substring match wins
    ("e4b", 4.0, None),
    ("qwen3.5-4b", 4.0, None),
    ("26b-a4b", 26.0, 4.0),
)


def params_b_for(model: str) -> tuple[float, float | None]:
    """(total, active) params in B from an obvious model name; else (DEFAULT_PARAMS_B, None)."""
    name = model.casefold()
    return next(((b, a) for key, b, a in KNOWN_PARAMS_B if key in name), (DEFAULT_PARAMS_B, None))


REASON_ACTIONS = ("reason.solve", "reason.check")
CAL_MAX_STEPS = 8  # a thinking model may probe the tool (schema, issue) before submitting
MAX_INGEST_BATCH = 500  # memory.ingest limit per call
ANSWER_MAX_CHARS = 200  # per-item answer kept in the result file
CAL_TOKEN_FACTOR = 2  # five probes need more reasoning budget than one puzzle

# Per-mode request budget: thinking models spend completion tokens (and time) on reasoning first.
MODES: dict[str, dict[str, Any]] = {
    "off": {"thinking": False, "max_tokens": 512, "timeout_s": 120.0},
    "on": {"thinking": True, "max_tokens": 2048, "timeout_s": 300.0},
}
ENV_CONCURRENCY = "CHRYSALIS_EVAL_CONCURRENCY"
ENV_FEEDBACK = "CHRYSALIS_EVAL_FEEDBACK"
ENV_ADVICE_DB = "CHRYSALIS_ADVICE_DB"
ENV_BIO_DEFAULTS = "CHRYSALIS_BIO_DEFAULTS"
FEEDBACK_ACTION = "reason.solve"  # what the advised arm's advice concerns (hard puzzles)
ADVICE_STATS_ROWS = 50  # rows kept in the compact meta.advice_stats snapshot
ENV_MAX_TOKENS = {"on": "CHRYSALIS_EVAL_MAX_TOKENS_ON", "off": "CHRYSALIS_EVAL_MAX_TOKENS_OFF"}
ENV_TIMEOUT = {"on": "CHRYSALIS_EVAL_TIMEOUT_ON", "off": "CHRYSALIS_EVAL_TIMEOUT_OFF"}
# Items of one arm in flight at once (set by ``_eval``; contextvars are inherited by asyncio tasks,
# so the ~15 ``_arm`` call sites need no extra parameter). 1 = the original sequential loop.
ARM_CONCURRENCY: contextvars.ContextVar[int] = contextvars.ContextVar("arm_concurrency", default=1)


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _env_positive_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds, got {raw!r}")
    return value


def feedback_enabled() -> bool:
    """``CHRYSALIS_EVAL_FEEDBACK=1``: the advised arm reports each outcome via meta.feedback."""
    return os.environ.get(ENV_FEEDBACK, "0").strip() == "1"


def bio_defaults_all() -> bool:
    return os.environ.get(ENV_BIO_DEFAULTS, "off").strip().lower() == "all"


def concurrency_from_env() -> int:
    return _env_positive_int(ENV_CONCURRENCY) or 1


def mode_config(mode: str) -> dict[str, Any]:
    """``MODES[mode]`` with ``max_tokens`` / ``timeout_s`` overridable by
    CHRYSALIS_EVAL_MAX_TOKENS_ON/_OFF and CHRYSALIS_EVAL_TIMEOUT_ON/_OFF."""
    cfg = dict(MODES[mode])
    cfg["max_tokens"] = _env_positive_int(ENV_MAX_TOKENS[mode]) or cfg["max_tokens"]
    cfg["timeout_s"] = _env_positive_float(ENV_TIMEOUT[mode]) or cfg["timeout_s"]
    return cfg


SYS_QA = "Answer with ONLY the answer phrase, no sentence."
SYS_QA_MEM = (
    SYS_QA + " Facts live in the `chrysalis` memory tool: call domain='memory', action='recall' "
    "with params {query, k:5, mode:'ppr'}; results/facts hold the answer. Answer from them."
)
SYS_QA_CTX = SYS_QA + " Answer using only these facts:\n{facts}"
SYS_PZ = (
    'Reply with ONLY a JSON object mapping each variable name to its integer value, e.g. {"x": 1}.'
)
SYS_PZ_SOLVER = (
    SYS_PZ + " You may call the `chrysalis` tool: domain='reason', action='solve', params "
    "{smt2: '(declare-const x Int)(assert ...)'} (SMT-LIB 2, no check-sat needed); "
    "use the model it returns."
)
DSL_DOC = (
    "domain='reason', action='solve', params {constraints: {vars: {x: 'int', y: 'int'}, "
    "assert: ['x + y == 10', 'x > y'], goal: 'sat'|'min:x'|'max:x'}}. Operators: + - * == != < "
    "<= > >= and or not, parentheses, integer literals. Every variable needs bounds in `assert`. "
    "The result holds the model; with goal min/max it is the optimum."
)
SYS_HARD_TOOL = (
    SYS_PZ + " You may call the `chrysalis` tool (a constraint solver): " + DSL_DOC + " Answer "
    "with the model it returns."
)
SYS_ADVISED_PREAMBLE = (
    "\n\nYou are {model} (family {family}, {params_desc} parameters, thinking "
    "{thinking_json}). The "
    "`chrysalis` tool description lists evidence-based advisory rules for a model like you "
    "(rec: recommended, dis: discouraged); each rule is followed by `In words:`, a plain-language "
    "hint you can check by counting the variables, the listed conditions and the min/max goal "
    "of THIS task. You may also call meta.handshake or meta.advise. Decide yourself "
    "whether the solver is worth using for THIS task."
)
# Conditional on purpose: an unconditional "ONLY the JSON object" reminder suppressed every tool
# call of E4B with thinking off (use 0.0/0.0 in the T17 ablations).
SYS_FORMAT_REMINDER = (
    "\n\nWhen you have the values (from the solver or by yourself), your final reply is ONLY the "
    'JSON object mapping each variable name to its integer value, e.g. {"x": 1}. No prose, no '
    "Markdown."
)


def identity(
    llm: LlmClient,
    params_b: float,
    family: str | None = None,
    active_params_b: float | None = None,
) -> dict[str, Any]:
    """Who the advised arm declares in its handshake (the harness knows the model it runs)."""
    who: dict[str, Any] = {
        "model": llm.model,
        "family": family or llm.model.split("-")[0].lower(),
        "params_b": params_b,
        "thinking": llm.thinking,
    }
    if active_params_b is not None:
        who["active_params_b"] = active_params_b
    return who


def advised_system(who: dict[str, Any], prompt_hint: str = "") -> str:
    """SYS_HARD_TOOL verbatim (as in the tool arm) + advisory paragraph + ``prompt_hint`` (from
    the handshake response, like a client that injects it) + format reminder last.

    The only difference vs the tool arm is the appended advisory part, so H-adv is not
    confounded by prompt layout (a buried format instruction made E4B answer in prose).
    """
    active = who.get("active_params_b")
    params_desc = f"{who['params_b']}B" + (f" total, {active}B active" if active else "")
    advisory = SYS_ADVISED_PREAMBLE.format(
        **who, params_desc=params_desc, thinking_json=json.dumps(who["thinking"])
    )
    hint = f"\n\n{prompt_hint}" if prompt_hint else ""
    return SYS_HARD_TOOL + advisory + hint + SYS_FORMAT_REMINDER


def context_facts() -> str:
    """ALL facts of the QA dataset, shuffled with a fixed seed, one sentence per line."""
    facts = [f"{s} {r.replace('_', ' ')} {o}." for i in QA_ITEMS for s, r, o in i.triples]
    random.Random(CONTEXT_SEED).shuffle(facts)
    return "\n".join(facts)


def _rate(runs: list[dict[str, Any]], hit: Callable[[dict[str, Any]], bool]) -> float | None:
    return round(sum(hit(r) for r in runs) / len(runs), 4) if runs else None


def _by_subset(
    runs: list[dict[str, Any]], correct: list[bool], tags: list[str] | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag in dict.fromkeys(tags or []):
        idx = [i for i, t in enumerate(tags or []) if t == tag and i < len(runs)]
        if idx:
            out[tag] = {
                "n": len(idx),
                "accuracy": round(sum(correct[i] for i in idx) / len(idx), 4),
                "reason_use_rate": _rate(
                    [runs[i] for i in idx], lambda r: any(a in REASON_ACTIONS for a in r["actions"])
                ),
            }
    return out


def _items(
    runs: list[dict[str, Any]],
    correct: list[bool],
    tags: list[str] | None,
    values: list[bool] | None = None,
    lenient: list[bool] | None = None,
) -> list[dict[str, Any]]:
    """Per-item record: what the model answered, which actions it took, what it cost."""
    out = []
    for i, run in enumerate(runs):
        item: dict[str, Any] = {"i": i}
        if tags and i < len(tags):
            item["tag"] = tags[i]
        item.update(
            correct=bool(correct[i]),
            **({"correct_lenient": bool(lenient[i])} if lenient else {}),
            **({"values_correct": bool(values[i])} if values else {}),
            status=run["status"],
            finish_reason=run.get("finish_reason"),
            actions=list(run.get("actions", [])),
            answer=(run.get("answer") or "")[:ANSWER_MAX_CHARS],
            prompt_tokens=run["metrics"]["llm_prompt_tokens"],
            completion_tokens=run["metrics"]["llm_completion_tokens"],
        )
        out.append(item)
    return out


def _summarize(
    runs: list[dict[str, Any]],
    correct: list[bool],
    tags: list[str] | None = None,
    values: list[bool] | None = None,
    lenient: list[bool] | None = None,
) -> dict[str, Any]:
    n = len(runs)
    sums = {k: sum(r["metrics"][k] for r in runs) for k in runs[0]["metrics"]} if runs else {}
    return {
        "n": n,
        "accuracy": round(sum(correct) / n, 4) if n else None,
        **({"values_accuracy": round(sum(values) / n, 4)} if values and n else {}),
        # QA arms only: strict OR (integer gold and exactly one equal number in the answer)
        **({"accuracy_lenient": round(sum(lenient) / n, 4)} if lenient and n else {}),
        # share of items whose completion hit max_tokens (answer possibly cut, not wrong per se)
        "truncated_rate": _rate(runs, lambda r: r.get("finish_reason") == "length"),
        "statuses": sorted({r["status"] for r in runs}),
        "mean_prompt_tokens": round(sums.get("llm_prompt_tokens", 0) / n, 1) if n else None,
        "mean_completion_tokens": round(sums.get("llm_completion_tokens", 0) / n, 1) if n else None,
        "mean_reasoning_chars": round(sums.get("reasoning_chars", 0) / n, 1) if n else None,
        "mean_tool_calls": round(sums.get("tool_calls", 0) / n, 2) if n else None,
        "mean_tool_bytes": round(sums.get("tool_bytes", 0) / n, 1) if n else None,
        "mean_wall_s": round(sums.get("wall_s", 0) / n, 2) if n else None,
        "reason_use_rate": _rate(
            runs, lambda r: any(a in REASON_ACTIONS for a in r.get("actions", []))
        ),
        "memory_use_rate": _rate(runs, lambda r: "memory.recall" in r.get("actions", [])),
        "handshake_first_rate": _rate(
            runs, lambda r: bool(r.get("actions")) and r["actions"][0] == "meta.handshake"
        ),
        **({"by_subset": _by_subset(runs, correct, tags)} if tags else {}),
        "items": _items(runs, correct, tags, values, lenient),
    }


async def _arm(
    llm: LlmClient,
    items: list[tuple[str, Any]],
    system: str | Callable[[str], str],
    tool: bool,
    score: Callable[[str | None, Any], bool],
    db: str | None,
    seed: Callable[[ChrysalisSession], Awaitable[None]] | None,
    max_tokens: int,
    declare: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    values_score: Callable[[str | None, Any], bool] | None = None,
    lenient_score: Callable[[str | None, Any], bool] | None = None,
    concurrency: int | None = None,
    features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One arm. ``concurrency`` (default: the ``ARM_CONCURRENCY`` context value) items run at once.

    Seeding and the handshake happen once, before any item. Results keep item order. After the
    first UNREACHABLE the not-yet-started items are skipped and the arm is cut at that item,
    exactly as the sequential loop stops. One shared ``ChrysalisSession`` serves all workers: its
    ``call`` is serialized by a lock, so the shared db is never touched concurrently.

    Closed loop (``declare`` and ``features`` given, ``CHRYSALIS_EVAL_FEEDBACK=1``): when an item
    ends, the harness sends meta.feedback (``used`` = the model called the solver, the item's task
    features, outcome from its score) through the same session, so meta.advise learns during the
    run. Items finish in a nondeterministic order when concurrent, so the order in which the
    evidence accrues is too; the totals are not.
    """
    runs: list[dict[str, Any]] = []
    golds: list[Any] = []
    correct: list[bool] = []
    values: list[bool] = []
    lenient: list[bool] = []
    width = max(1, concurrency or ARM_CONCURRENCY.get())
    loop = bool(declare and features and feedback_enabled())
    sent = 0
    snapshot: dict[str, Any] | None = None
    async with ChrysalisSession(db) as cs:
        if seed:
            await seed(cs)
        prompt_hint = ""
        if declare:  # the harness (like a client config) declares the model: rules reach the
            shake = json.loads(  # tool description; prompt_hint goes into the system prompt
                await cs.call(domain="meta", action="handshake", params=declare)
            )
            prompt_hint = shake.get("prompt_hint", "")
        sys_prompt = system(prompt_hint) if callable(system) else system
        gate = asyncio.Semaphore(width)
        down = asyncio.Event()  # set by the first UNREACHABLE

        async def one(index: int, question: str, gold: Any) -> dict[str, Any] | None:
            nonlocal sent
            async with gate:
                if down.is_set():
                    return None
                run = await run_agent(
                    llm, question, cs if tool else None, system=sys_prompt, max_tokens=max_tokens
                )
                if run["status"] == "UNREACHABLE":
                    down.set()
                elif loop:
                    won = run["status"] == "OK" and score(run["answer"], gold)
                    reply = json.loads(
                        await cs.call(
                            domain="meta",
                            action="feedback",
                            params={
                                "action": FEEDBACK_ACTION,
                                "used": any(a in REASON_ACTIONS for a in run.get("actions", [])),
                                "task": (features or [])[index],
                                "outcome": "success" if won else "failure",
                            },
                        )
                    )
                    sent += reply.get("status") == "OK"
                return run

        outcomes = await asyncio.gather(*(one(i, q, g) for i, (q, g) in enumerate(items)))
        if loop:
            stats = json.loads(
                await cs.call(
                    domain="meta",
                    action="advice_stats",
                    params={"action": FEEDBACK_ACTION, "limit": ADVICE_STATS_ROWS},
                )
            )
            snapshot = {k: stats.get(k) for k in ("model", "total", "rows", "truncated")}
        for (_, gold), run in zip(items, outcomes, strict=True):
            if run is None:
                break
            runs.append(run)
            golds.append(gold)
            if run["status"] == "UNREACHABLE":
                break  # same cut as the sequential loop; later finished items are discarded
    for run, gold in zip(runs, golds, strict=True):
        correct.append(run["status"] == "OK" and score(run["answer"], gold))
        if values_score:  # diagnostic on the FULL answer (before ANSWER_MAX_CHARS truncation)
            values.append(run["status"] == "OK" and values_score(run["answer"], gold))
        if lenient_score:
            lenient.append(run["status"] == "OK" and lenient_score(run["answer"], gold))
    summary = _summarize(
        runs,
        correct,
        tags,
        values if values_score else None,
        lenient if lenient_score else None,
    )
    # arm flag: the advised arm injected the handshake's prompt_hint into its system prompt
    if not declare:
        return summary
    extra: dict[str, Any] = {"prompt_hint_used": bool(prompt_hint)}
    if loop:
        extra |= {"feedback_sent": sent, "advice_stats": snapshot}
    return {**summary, **extra}


def probe_prompt(probes: list[dict[str, str]]) -> str:
    lines = "\n".join(f"- {p['id']} ({p['family']}): {p['prompt']}" for p in probes)
    return (
        "Self-assessment. Solve each probe, then submit ALL answers in ONE call to the "
        "`chrysalis` tool: domain='meta', action='calibrate', params {answers: {'<probe id>': "
        "<answer>, ...}} (keys are the bare probe ids such as 'reason-1'; answer = a string, or a "
        "JSON list of strings where the probe asks for a list). Probes:\n" + lines
    )


async def calibrate_once(llm: LlmClient, max_tokens: int, who: dict[str, Any]) -> dict[str, Any]:
    """The bridge issues ``meta.calibrate``; the model answers via the tool; scores read back."""
    async with ChrysalisSession(None) as cs:
        issued = json.loads(await cs.call(domain="meta", action="calibrate", verbosity="summary"))
        if issued.get("status") != "OK":
            return {"status": "ERROR", "error": issued.get("error", "calibrate not issued")}
        await cs.call(domain="meta", action="handshake", params=who)
        run = await run_agent(
            llm,
            probe_prompt(issued["probes"]),
            cs,
            system="Answer carefully; use the tool only to submit the answers.",
            max_steps=CAL_MAX_STEPS,
            max_tokens=max_tokens * CAL_TOKEN_FACTOR,
        )
        out = json.loads(await cs.call(domain="meta", action="discover", params={"domain": "meta"}))
        scores = out.get("profile", {}).get("scores", {})
        submitted = "meta.calibrate" in run.get("actions", [])
    # an empty closing message is fine once the answers were submitted through the tool
    status = "OK" if run["status"] == "EMPTY" and submitted else run["status"]
    return {
        "status": status,
        "scores": scores,
        "submitted": submitted,
        "actions": run.get("actions", []),  # trace: why a run ends without scores is readable
        "error": run.get("error"),
        "prompt_tokens": run["metrics"]["llm_prompt_tokens"],
        "completion_tokens": run["metrics"]["llm_completion_tokens"],
        "wall_s": run["metrics"]["wall_s"],
    }


def server_ctx() -> int:
    """Window of the local server (same env var ``validation.llm_server`` launches it with)."""
    try:
        return int(os.environ.get(SERVER_CTX_ENV, DEFAULT_SERVER_CTX))
    except ValueError:
        return DEFAULT_SERVER_CTX


async def _scale_group(llm: LlmClient, mt: int, limit: int | None, tmp: str) -> dict[str, Any]:
    """H-a-scale: more facts than the window. Context arm: the tail that fits; memory: all."""
    data = scaled_qa()
    items = data.items[:limit] if limit else data.items
    kept, dropped = fit_context(data.facts, server_ctx() - SCALE_RESERVE_TOKENS)

    async def seed_all(cs: ChrysalisSession) -> None:
        rows = [{"s": s, "r": r, "o": o} for s, r, o in data.triples]
        for i in range(0, len(rows), MAX_INGEST_BATCH):
            out = json.loads(
                await cs.call(
                    domain="memory",
                    action="ingest",
                    params={"triples": rows[i : i + MAX_INGEST_BATCH]},
                )
            )
            if out.get("status") != "OK":
                raise RuntimeError(f"scale ingest failed: {out.get('error')}")

    qa = [(i.question, i.answer) for i in items]
    system = SYS_QA_CTX.format(facts="\n".join(kept))
    ctx_arm = await _arm(
        llm, qa, system, False, score_qa, None, None, mt, lenient_score=score_qa_lenient
    )
    ctx_arm.update(
        total_facts=len(data.facts),
        facts_in_context=len(kept),
        dropped_fraction=dropped,
        answerable_in_context=answerable_in(tuple(items), kept),
    )
    db = str(Path(tmp) / "scale.db")
    mem_arm = await _arm(
        llm, qa, SYS_QA_MEM, True, score_qa, db, seed_all, mt, lenient_score=score_qa_lenient
    )
    return {"context": ctx_arm, "memory": mem_arm}


async def _hbe2e_group(llm: LlmClient, mt: int, limit: int | None, tmp: str) -> dict[str, Any]:
    """H-b-e2e: per arm the harness replays the same learning operations (KB, teach A, teach B,
    interference) over stdio; the model then answers the set-A ``test`` questions with memory."""
    data = alias_dataset()
    items = data.set_a[:limit] if limit else data.set_a
    qa = [(i.test, i.answer) for i in items]
    arms: dict[str, Any] = {}
    for arm in HBE2E_ARMS:

        async def seed(cs: ChrysalisSession, arm: str = arm) -> None:
            for action, params in prepare_ops(arm, data):
                out = json.loads(await cs.call(domain="memory", action=action, params=params))
                if out.get("status") != "OK":
                    raise RuntimeError(f"hbe2e {arm} {action} failed: {out.get('error')}")
            if bio_defaults_all() and arm != "no_learn":
                # learning phase over, test phase next: one sleep consolidation (the server fills
                # replay/usage_weighting/capture from CHRYSALIS_BIO_DEFAULTS=all)
                out = json.loads(await cs.call(domain="memory", action="consolidate", params={}))
                if out.get("status") != "OK":
                    raise RuntimeError(f"hbe2e {arm} consolidate failed: {out.get('error')}")

        db = str(Path(tmp) / f"hbe2e_{arm}_{int(llm.thinking)}.db")  # fresh per mode
        arms[arm] = await _arm(
            llm, qa, SYS_QA_MEM, True, score_qa, db, seed, mt, lenient_score=score_qa_lenient
        )
    return arms


def _features(puzzle: Any) -> dict[str, Any]:
    """The task features meta.advise would compute for the puzzle's reference reason.solve call."""
    return task_features.derive(
        task_features.extract(FEEDBACK_ACTION, {"constraints": puzzle.constraints})
    )


async def _group(
    name: str,
    client: LlmClient,
    mt: int,
    limit: int | None,
    params_b: float,
    tmp: str,
    active_params_b: float | None = None,
) -> dict[str, Any]:
    cut = (lambda xs: xs[:limit]) if limit else (lambda xs: xs)
    qa = cut([(i.question, i.answer) for i in QA_ITEMS])
    pz = cut([(p.text, p.answer) for p in PUZZLES])
    hard = cut([(p.text, p.answer) for p in HARD_PUZZLES])

    async def seed_facts(cs: ChrysalisSession) -> None:
        triples = [{"s": s, "r": r, "o": o} for i in QA_ITEMS for s, r, o in i.triples]
        await cs.call(domain="memory", action="ingest", params={"triples": triples})

    if name == "ha":
        db = str(Path(tmp) / "eval.db")
        return {
            "alone": await _arm(
                client, qa, SYS_QA, False, score_qa, None, None, mt, lenient_score=score_qa_lenient
            ),
            "memory": await _arm(
                client,
                qa,
                SYS_QA_MEM,
                True,
                score_qa,
                db,
                seed_facts,
                mt,
                lenient_score=score_qa_lenient,
            ),
        }
    if name == "ctx":
        system = SYS_QA_CTX.format(facts=context_facts())
        return {
            "context": await _arm(
                client, qa, system, False, score_qa, None, None, mt, lenient_score=score_qa_lenient
            )
        }
    if name == "hc":
        return {
            "alone": await _arm(client, pz, SYS_PZ, False, score_puzzle, None, None, mt),
            "reason": await _arm(client, pz, SYS_PZ_SOLVER, True, score_puzzle, None, None, mt),
        }
    if name == "hascale":
        return await _scale_group(client, mt, limit, tmp)
    if name == "hbe2e":
        return await _hbe2e_group(client, mt, limit, tmp)
    if name == "hard":
        return {
            "alone": await _arm(
                client,
                hard,
                SYS_PZ,
                False,
                score_puzzle,
                None,
                None,
                mt,
                values_score=score_puzzle_values,
            ),
            "tool": await _arm(
                client,
                hard,
                SYS_HARD_TOOL,
                True,
                score_puzzle,
                None,
                None,
                mt,
                values_score=score_puzzle_values,
            ),
        }
    if name == "adv":
        who = identity(client, params_b, active_params_b=active_params_b)
        adv_features = [_features(p) for p in cut(list(HARD_PUZZLES))]
        arm = await _arm(
            client,
            hard,
            lambda hint: advised_system(who, hint),
            True,
            score_puzzle,
            None,
            None,
            mt,
            who,
            values_score=score_puzzle_values,
            features=adv_features,
        )
        return {"advised": {**arm, "in_sample": True}}  # thresholds were fitted on HARD_PUZZLES
    if name == "heldout":
        who = identity(client, params_b, active_params_b=active_params_b)
        hh, he = (
            cut([(p.text, p.answer) for p in HELDOUT_HARD]),
            cut([(p.text, p.answer) for p in HELDOUT_EASY]),
        )
        items, tags = [*hh, *he], ["hard"] * len(hh) + ["easy"] * len(he)
        held_features = [_features(p) for p in (*cut(list(HELDOUT_HARD)), *cut(list(HELDOUT_EASY)))]
        return {
            "alone": await _arm(
                client,
                items,
                SYS_PZ,
                False,
                score_puzzle,
                None,
                None,
                mt,
                values_score=score_puzzle_values,
            ),
            "tool": await _arm(
                client,
                items,
                SYS_HARD_TOOL,
                True,
                score_puzzle,
                None,
                None,
                mt,
                values_score=score_puzzle_values,
            ),
            "advised": await _arm(
                client,
                items,
                lambda hint: advised_system(who, hint),
                True,
                score_puzzle,
                None,
                None,
                mt,
                who,
                tags,
                values_score=score_puzzle_values,
                features=held_features,
            ),
        }
    if name == "cal":
        return await calibrate_once(
            client, mt, identity(client, params_b, active_params_b=active_params_b)
        )
    raise ValueError(f"unknown group {name!r}")


@contextlib.contextmanager
def _advice_db_scope() -> Iterator[None]:
    """Keep the learned advice of this run under the results dir, never in the user's real db.

    Unless ``CHRYSALIS_ADVICE_DB`` is already set, point it at a fresh per-run file so runs (and
    models) cannot contaminate each other; the stdio servers inherit the environment.
    """
    if os.environ.get(ENV_ADVICE_DB):
        yield
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    os.environ[ENV_ADVICE_DB] = str(RESULTS_DIR / f"advice_{stamp}.db")
    try:
        yield
    finally:
        os.environ.pop(ENV_ADVICE_DB, None)


async def _eval(
    llm: LlmClient,
    groups: tuple[str, ...],
    modes: tuple[str, ...],
    limit: int | None,
    params_b: float,
    active_params_b: float | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    ARM_CONCURRENCY.set(concurrency)
    with _advice_db_scope(), tempfile.TemporaryDirectory() as tmp:
        for mode in modes:
            cfg = mode_config(mode)
            client = llm.derive(cfg["thinking"], cfg["timeout_s"])
            results[mode] = {
                "config": {
                    **cfg,
                    "concurrency": concurrency,
                    "bio_defaults": "all" if bio_defaults_all() else "off",
                    "feedback": feedback_enabled(),
                }
            }
            for g in groups:
                if g in LONG_GROUPS and mode not in HA_MODES:
                    continue
                results[mode][GROUP_NAMES[g]] = await _group(
                    g, client, cfg["max_tokens"], limit, params_b, tmp, active_params_b
                )
    return results


def run(
    llm: LlmClient | None = None,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    modes: tuple[str, ...] = tuple(MODES),
    limit: int | None = None,
    params_b: float | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    llm = llm or LlmClient()
    concurrency = concurrency or concurrency_from_env()
    active_params_b: float | None = None
    if params_b is None:
        params_b, active_params_b = params_b_for(llm.model)
    unknown = [g for g in groups if g not in ALL_GROUPS] + [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"unknown group/mode: {unknown}")
    base = {
        "hypothesis": "H-a/H-c",
        "config": {
            "model": llm.model,
            "url": llm.base_url,
            "params_b": params_b,
            "active_params_b": active_params_b,
        },
    }
    if not llm.available():
        return {**base, "status": "UNREACHABLE", "data": {}}
    results = asyncio.run(_eval(llm, groups, modes, limit, params_b, active_params_b, concurrency))
    primary = results[modes[0]]
    data = {
        k: v
        for k, v in primary.items()
        if k not in ("config", GROUP_NAMES["cal"]) and k in {GROUP_NAMES[g] for g in groups}
    }
    return {**base, "status": "OK", "data": data, "modes": results}


def merge(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``new`` on ``old`` per mode and group (an UNREACHABLE run never erases data)."""
    if new.get("status") != "OK":  # a non-OK run never erases OK data
        return old if old and old.get("status") == "OK" else new
    if not old or old.get("status") != "OK":
        return new
    modes = {m: {**v} for m, v in old.get("modes", {}).items()}
    for mode, groups in new.get("modes", {}).items():
        modes[mode] = {**modes.get(mode, {}), **groups}
    primary = modes.get("off") or next(iter(modes.values()), {})
    data = {k: v for k, v in primary.items() if k not in ("config", GROUP_NAMES["cal"])}
    return {**old, **new, "data": data, "modes": modes}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m validation.eval_llm", description=__doc__)
    ap.add_argument("--hyp", default=",".join(DEFAULT_GROUPS), help=f"{ALL_GROUPS} or 'all'")
    ap.add_argument("--thinking", choices=("off", "on", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None, help="items per dataset (smoke runs)")
    ap.add_argument("--params-b", type=float, default=None, help="default: from the served model")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"items of an arm in flight at once (env {ENV_CONCURRENCY}; default 1). Match the "
        "server's CHRYSALIS_LLM_PARALLEL slots",
    )
    args = ap.parse_args(argv)
    if args.concurrency is not None and args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    groups = ALL_GROUPS if args.hyp == "all" else tuple(g.strip() for g in args.hyp.split(","))
    modes = tuple(MODES) if args.thinking == "both" else (args.thinking,)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(
        groups=groups,
        modes=modes,
        limit=args.limit,
        params_b=args.params_b,
        concurrency=args.concurrency,
    )
    path = RESULTS_DIR / RESULT_FILE
    old = json.loads(path.read_text()) if path.exists() else None
    if result.get("status") != "OK" and old and old.get("status") == "OK":
        print(f"NOTICE: {result.get('status')} run; kept existing OK data in {path} (not written)")
    else:
        write(merge(old, result))
    print(result["status"], json.dumps(result.get("data", {}))[:300])


def write(result: dict[str, Any]) -> None:
    """``llm.json`` (combined) plus one ``llm_thinking_<mode>.json`` per mode."""
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(result, indent=2))
    for name, data in result.get("modes", {}).items():
        per_mode = {k: result[k] for k in ("hypothesis", "config", "status")}
        (RESULTS_DIR / f"llm_thinking_{name}.json").write_text(
            json.dumps({**per_mode, "thinking": name, "data": data}, indent=2)
        )


if __name__ == "__main__":
    main()
