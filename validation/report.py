"""Merge validation/results/*.json into validation/results/RESULTS.md with a verdict per hypothesis.

Verdicts: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE (missing or unreachable data), with numbers.
"""

import json
from pathlib import Path
from typing import Any

from validation import REPORT_PATH, RESULTS_DIR
from validation.bench_ha_prime import setting_key
from validation.bench_hb import PRECONDITION_MIN
from validation.bench_hb_e2e import RESULT_FILE as HB_E2E_FILE

RECALL_RATIO_MIN = 0.9  # H-a': fly cluster precision must reach 90% of exact cosine's
CTX_TOLERANCE = 0.05  # H-a-ctx: memory may trail the full-context baseline by this accuracy
HARD_GAIN_MIN = 0.1  # H-c-hard: tool - alone
ADV_TOLERANCE = 0.05  # H-adv: advised may trail max(alone, tool) by this accuracy
HBE2E_GAIN_MIN = 0.2  # H-b-e2e: fly_update - no_learn LLM accuracy, in every mode run
HBE2E_RETENTION_MIN = 0.9  # H-b-e2e: retained recall@k / taught recall@k of fly_update
SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"


def _load(results_dir: Path, name: str) -> dict[str, Any] | None:
    path = results_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def verdict_ha_prime(res: dict[str, Any] | None) -> tuple[str, str]:
    """Judged on the shipped FlyHash parameters; the best swept setting is reported alongside."""
    if not res:
        return INCONCLUSIVE, "no data"
    default = res["config"]["flyhash_default"]
    key = setting_key(default["k"], default["c"])
    rows = res["data"].values()
    worst = min(r["fly"][key]["fly_over_cosine"] for r in rows)
    smaller = all(r["fly"][key]["bytes_per_node"] < r["cosine"]["bytes_per_node"] for r in rows)
    best_key, best = max(
        ((k, min(r["fly"][k]["fly_over_cosine"] for r in rows)) for k in next(iter(rows))["fly"]),
        key=lambda kv: kv[1],
    )
    ok = worst >= RECALL_RATIO_MIN and smaller
    return (SUPPORTED if ok else NOT_SUPPORTED), (
        f"fly/cosine cluster precision at shipped {key}: worst {worst:.3f} "
        f"(need >= {RECALL_RATIO_MIN}); fly bytes/node smaller: {smaller}; "
        f"best swept setting {best_key}: {best:.3f}"
    )


def verdict_hb(res: dict[str, Any] | None) -> tuple[str, str]:
    """Sparse must learn (precondition) and end strictly above every baseline that also learned."""
    if not res:
        return INCONCLUSIVE, "no data"
    arms = res["data"]
    sparse = arms["sparse"]
    last = max(sparse, key=int)
    valid = {a: v for a, v in arms.items() if a != "sparse" and v["0"] >= PRECONDITION_MIN}
    skipped = sorted(set(arms) - set(valid) - {"sparse"})
    detail = "; ".join(
        f"{a} N=0 {v['0']:.2f} -> N={last} {v[last]:.2f} (retention "
        f"{v[last] / v['0'] if v['0'] else 0:.2f})"
        for a, v in arms.items()
    )
    note = f"{detail}; baselines that failed the N=0 >= {PRECONDITION_MIN} precondition: {skipped}"
    if sparse["0"] < PRECONDITION_MIN:
        return INCONCLUSIVE, f"sparse arm did not learn the old associations; {note}"
    if not valid:
        return INCONCLUSIVE, f"no baseline learned the old associations; {note}"
    best = max(v[last] for v in valid.values())
    return (SUPPORTED if sparse[last] > best else NOT_SUPPORTED), note


def _hbe2e_arm_note(arms: dict[str, Any], name: str) -> str:
    arm = arms.get(name)
    if not arm:
        return ""
    return f"{name} {arm['accuracy']} (memory use {arm.get('memory_use_rate')})"


def verdict_hb_e2e(hb: dict[str, Any] | None, llm: dict[str, Any] | None) -> tuple[str, str]:
    """H-b-e2e: with the model in the loop, ``update`` must add >= HBE2E_GAIN_MIN accuracy over
    no learning in EVERY mode run, and keep >= HBE2E_RETENTION_MIN of its recall@k after
    interference.

    The comparison with the alias-ingest baseline is informational and never changes the verdict.
    Retrieval-only results render (as INCONCLUSIVE with the numbers) without LLM results.
    """
    fly = (hb or {}).get("data", {}).get("fly_update")
    retr = ""
    if fly:
        retr = f"retrieval recall@{hb['config']['k']} (fly_update): taught {fly['taught']} -> " + (
            f"retained {fly['retained']} (retention "
            f"{fly['retention'] if fly['retention'] is not None else 'n/a'}); "
        )
        retr += "; ".join(
            f"{a} {v['taught']} -> {v['retained']}"
            for a, v in hb["data"].items()
            if a != "fly_update"
        )
    parts, gains = [], []
    for mode in ("off", "on"):
        arms = _group(llm, mode, "H-b-e2e")
        if not arms or "fly_update" not in arms or "no_learn" not in arms:
            continue
        gain = round(arms["fly_update"]["accuracy"] - arms["no_learn"]["accuracy"], 4)
        gains.append(gain >= HBE2E_GAIN_MIN)
        notes = (_hbe2e_arm_note(arms, a) for a in ("alias_ingest", "alias_ingest_clean"))
        info = "; ".join(n for n in notes if n)
        parts.append(
            f"thinking {mode}: no_learn {arms['no_learn']['accuracy']} -> fly_update "
            f"{arms['fly_update']['accuracy']} (gain {gain:+}, memory use "
            f"{arms['fly_update'].get('memory_use_rate')}); informational vs baseline: "
            f"{info or '-'}"
        )
    llm_text = "; ".join(parts) + f" (need gain >= {HBE2E_GAIN_MIN} in every mode run)"
    if not fly:
        return INCONCLUSIVE, "no retrieval data" + (f"; {llm_text}" if parts else "")
    kept = fly["taught"] > 0 and fly["retained"] >= HBE2E_RETENTION_MIN * fly["taught"]
    retention = f"{retr} (need retention >= {HBE2E_RETENTION_MIN}: {kept})"
    if not gains:
        return INCONCLUSIVE, f"LLM arms not run; {retention}"
    ok = all(gains) and kept
    return (SUPPORTED if ok else NOT_SUPPORTED), f"{llm_text}; {retention}"


def verdict_hmcp(res: dict[str, Any] | None) -> tuple[str, str]:
    if not res:
        return INCONCLUSIVE, "no data"
    d = res["data"]
    ok = d["batch"]["est_tokens"] < d["sequential"]["est_tokens"] <= d["granular"]["est_tokens"]
    return (SUPPORTED if ok else NOT_SUPPORTED), (
        f"est_tokens batch {d['batch']['est_tokens']} < sequential {d['sequential']['est_tokens']} "
        f"<= granular {d['granular']['est_tokens']} (granular SIMULATED): {ok}"
    )


def verdict_ha(res: dict[str, Any] | None, llm: dict[str, Any] | None) -> tuple[str, str]:
    """PPR must beat plain seeds, match the 2-round iterative baseline, at no higher cost."""
    if not res or "iterative" not in res["data"]:
        return INCONCLUSIVE, "no data"
    seeds, it, ppr = (res["data"][m] for m in ("seeds", "iterative", "ppr"))
    better = ppr["recall@5"] > seeds["recall@5"] and ppr["recall@5"] >= it["recall@5"]
    cheaper = ppr["query_ms"] <= it["query_ms"] and ppr["response_bytes"] <= it["response_bytes"]
    note = (
        f"recall@5 ppr {ppr['recall@5']:.3f} vs seeds {seeds['recall@5']:.3f} vs iterative "
        f"{it['recall@5']:.3f}; ppr (1 call) {ppr['query_ms']} ms / {ppr['response_bytes']} B vs "
        f"iterative (2 calls) {it['query_ms']} ms / {it['response_bytes']} B "
        f"(not costlier than iterative: {cheaper})"
    )
    return (SUPPORTED if better and cheaper else NOT_SUPPORTED), note


def verdict_llm(llm: dict[str, Any] | None, key: str, with_arm: str) -> tuple[str, str]:
    if not llm or llm.get("status") != "OK":
        return INCONCLUSIVE, "live model unreachable / not run"
    arms = llm["data"].get(key)
    if not arms or with_arm not in arms:
        return INCONCLUSIVE, f"{key} not run"
    a, w = arms["alone"], arms[with_arm]
    return (SUPPORTED if w["accuracy"] > a["accuracy"] else NOT_SUPPORTED), (
        f"accuracy alone {a['accuracy']} vs with {with_arm} {w['accuracy']} "
        f"(prompt tokens/q {a['mean_prompt_tokens']} -> {w['mean_prompt_tokens']})"
    )


def _group(llm: dict[str, Any] | None, mode: str, group: str) -> dict[str, Any] | None:
    if not llm or llm.get("status") != "OK":
        return None
    return llm.get("modes", {}).get(mode, {}).get(group) or None


def verdict_ctx(llm: dict[str, Any] | None) -> tuple[str, str]:
    """H-a-ctx: memory is a cheaper equivalent of the whole-dataset-in-prompt baseline."""
    mem = (_group(llm, "off", "H-a") or {}).get("memory")
    ctx = (_group(llm, "off", "H-a-ctx") or {}).get("context")
    if not mem or not ctx:
        return INCONCLUSIVE, "H-a memory and H-a-ctx context arms not both run"
    fewer = mem["mean_prompt_tokens"] < ctx["mean_prompt_tokens"]
    ok = mem["accuracy"] >= ctx["accuracy"] - CTX_TOLERANCE and fewer
    return (SUPPORTED if ok else NOT_SUPPORTED), (
        f"accuracy memory {mem['accuracy']} vs context {ctx['accuracy']} (tolerance "
        f"{CTX_TOLERANCE}); prompt tok/q {mem['mean_prompt_tokens']} vs "
        f"{ctx['mean_prompt_tokens']}; completion tok/q {mem['mean_completion_tokens']} vs "
        f"{ctx['mean_completion_tokens']}"
    )


SCALE_GAIN_MIN = 0.1  # H-a-scale: memory must beat the truncated context by this much


def verdict_scale(llm: dict[str, Any] | None) -> tuple[str, str]:
    """H-a-scale: with more facts than the window, memory beats the (truncated) context arm."""
    arms = _group(llm, "off", "H-a-scale")
    if not arms or "memory" not in arms or "context" not in arms:
        return INCONCLUSIVE, "H-a-scale context and memory arms not both run"
    mem, ctx = arms["memory"], arms["context"]
    gain = round(mem["accuracy"] - ctx["accuracy"], 4)
    return (SUPPORTED if gain >= SCALE_GAIN_MIN else NOT_SUPPORTED), (
        f"accuracy memory {mem['accuracy']} vs context {ctx['accuracy']} (gain {gain:+}, need >= "
        f"{SCALE_GAIN_MIN}); {ctx.get('facts_in_context')}/{ctx.get('total_facts')} facts fit "
        f"(dropped {ctx.get('dropped_fraction')}, answerable in context "
        f"{ctx.get('answerable_in_context')}); prompt tok/q {mem['mean_prompt_tokens']} vs "
        f"{ctx['mean_prompt_tokens']}"
    )


def verdict_hard(llm: dict[str, Any] | None) -> tuple[str, str]:
    """H-c-hard: the solver tool lifts accuracy on the hard set by >= HARD_GAIN_MIN."""
    parts, verdicts = [], []
    for mode in ("off", "on"):
        arms = _group(llm, mode, "H-c-hard")
        if not arms or "alone" not in arms or "tool" not in arms:
            continue
        gain = round(arms["tool"]["accuracy"] - arms["alone"]["accuracy"], 4)
        verdicts.append(gain >= HARD_GAIN_MIN)
        parts.append(
            f"thinking {mode}: alone {arms['alone']['accuracy']} -> tool "
            f"{arms['tool']['accuracy']} (gain {gain:+}, "
            f"tool use {arms['tool']['reason_use_rate']})" + _values_note(arms, ("alone", "tool"))
        )
    if not verdicts:
        return INCONCLUSIVE, "H-c-hard alone/tool arms not run"
    return (SUPPORTED if all(verdicts) else NOT_SUPPORTED), (
        "; ".join(parts) + f" (need gain >= {HARD_GAIN_MIN} in every mode run)"
    )


def _values_note(arms: dict[str, Any], names: tuple[str, ...]) -> str:
    """Diagnostic only (format-lenient accuracy); empty for old results without the key."""
    got = [
        f"{n} {arms[n]['values_accuracy']}"
        for n in names
        if arms.get(n, {}).get("values_accuracy") is not None
    ]
    return f" [values accuracy, format-lenient, diagnostic: {', '.join(got)}]" if got else ""


IN_SAMPLE_NOTE = "(in-sample; thresholds fitted on this set)"


def _subset_use(arm: dict[str, Any]) -> str:
    sub = arm.get("by_subset") or {}
    return ", ".join(
        f"{k} n={v['n']} acc {v['accuracy']} use {v['reason_use_rate']}" for k, v in sub.items()
    )


def verdict_adv(llm: dict[str, Any] | None) -> tuple[str, str]:
    """H-adv: advised accuracy within ADV_TOLERANCE of the better of alone and tool, same item set.

    The held-out group (``H-adv-heldout``: alone, tool and advised on items never used to fit the
    thresholds) is the only basis for SUPPORTED. A result on HARD_PUZZLES (``H-adv``; legacy runs
    carry no flag) is in-sample: it can refute but never support.
    """
    parts, verdicts, in_sample = [], [], False
    for mode in ("off", "on"):
        held = _group(llm, mode, "H-adv-heldout")
        if held and {"alone", "tool", "advised"} <= held.keys():
            arms, adv, tag = held, held["advised"], "held-out"
        else:
            arms = _group(llm, mode, "H-c-hard")
            adv = (_group(llm, mode, "H-adv") or {}).get("advised")
            if not arms or not adv or "tool" not in arms:
                continue
            tag = "HARD_PUZZLES"
            in_sample = in_sample or adv.get("in_sample", True)
        best = max(arms["alone"]["accuracy"], arms["tool"]["accuracy"])
        verdicts.append(adv["accuracy"] >= best - ADV_TOLERANCE)
        parts.append(
            f"thinking {mode} [{tag}]: advised {adv['accuracy']} vs best-of(alone, tool) {best} "
            f"(handshake first {adv['handshake_first_rate']}, tool use {adv['reason_use_rate']}"
            + (f"; {_subset_use(adv)}" if adv.get("by_subset") else "")
            + ")"
            + _values_note({**arms, "advised": adv}, ("alone", "tool", "advised"))
        )
    if not verdicts:
        return INCONCLUSIVE, "H-adv-heldout (or H-adv with H-c-hard) arms not run"
    text = "; ".join(parts) + f" (tolerance {ADV_TOLERANCE})"
    if in_sample:
        text += " " + IN_SAMPLE_NOTE
        return (INCONCLUSIVE if all(verdicts) else NOT_SUPPORTED), text
    return (SUPPORTED if all(verdicts) else NOT_SUPPORTED), text


def calibration_rows(llm: dict[str, Any] | None) -> list[list[Any]]:
    """Per mode and family: calibration score vs measured tool-minus-alone delta, sign agreement."""
    rows: list[list[Any]] = []
    for mode in ("off", "on"):
        cal = _group(llm, mode, "calibration")
        if not cal:
            continue
        for family, group, arm in (("reason", "H-c-hard", "tool"), ("memory", "H-a", "memory")):
            score = cal.get("scores", {}).get(family)
            arms = _group(llm, mode, group)
            if score is None or not arms or arm not in arms:
                rows.append([mode, family, score, "-", "-"])
                continue
            delta = round(arms[arm]["accuracy"] - arms["alone"]["accuracy"], 4)
            rows.append([mode, family, score, f"{delta:+}", (score >= 0.8) == (delta > 0)])
    return rows


def _llm_rows(llm: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for mode in ("off", "on"):
        for group, arms in llm.get("modes", {}).get(mode, {}).items():
            if group not in ("config", "calibration"):
                rows.append((f"{group} (thinking {mode})", arms))
    return rows or [(f"{h} (thinking off)", arms) for h, arms in llm["data"].items()]


MANUAL_MARKER = "<!-- manual:start -->"


def split_manual(text: str) -> str:
    """Everything after the marker line, verbatim ('' when the marker is absent)."""
    _, sep, tail = text.partition(MANUAL_MARKER + "\n")
    return tail if sep else ""


def render(results_dir: Path = RESULTS_DIR) -> str:
    ha_p, hb, hmcp, ha, llm, hb_e2e = (
        _load(results_dir, f)
        for f in ("ha_prime.json", "hb.json", "hmcp.json", "ha.json", "llm.json", HB_E2E_FILE)
    )
    verdicts = [
        ("H-a (retrieval)", *verdict_ha(ha, llm)),
        ("H-a (end-to-end LLM)", *verdict_llm(llm, "H-a", "memory")),
        ("H-a'", *verdict_ha_prime(ha_p)),
        ("H-b", *verdict_hb(hb)),
        ("H-b-e2e (learned aliases, model in the loop)", *verdict_hb_e2e(hb_e2e, llm)),
        ("H-a-ctx (memory vs full context)", *verdict_ctx(llm)),
        ("H-a-scale (facts beyond the window)", *verdict_scale(llm)),
        ("H-c (end-to-end LLM)", *verdict_llm(llm, "H-c", "reason")),
        ("H-c-hard (DSL solver, hard set)", *verdict_hard(llm)),
        ("H-adv (advised arm)", *verdict_adv(llm)),
        ("H-mcp", *verdict_hmcp(hmcp)),
    ]
    out = [
        "# Validation results (I4-M4)",
        "",
        "Generated by `validation/report.py` from `validation/results/*.json`. "
        "Do not edit above the manual marker.",
        "",
        "## Verdicts",
        "",
        _table(["Hypothesis", "Verdict", "Evidence"], [list(v) for v in verdicts]),
    ]
    if ha_p:
        rows = []
        for n, r in ha_p["data"].items():
            c = r["cosine"]
            rows.append([n, "cosine", "-", c["precision"], 1.0, c["bytes_per_node"], c["query_ms"]])
            rows += [
                [
                    n,
                    "flyhash",
                    k,
                    v["precision"],
                    v["fly_over_cosine"],
                    v["bytes_per_node"],
                    v["query_ms"],
                ]
                for k, v in r["fly"].items()
            ]
        out += [
            "",
            "## H-a' FlyHash vs exact cosine (cluster precision, sweep over k and c)",
            "",
            _table(["N", "method", "setting", "precision", "x cosine", "B/node", "ms"], rows),
        ]
    if ha:
        rows = [
            [
                "seeds (cosine since T11)" if m == "seeds" else m,
                v["recall@1"],
                v["recall@5"],
                v["query_ms"],
                v["response_bytes"],
                v["tool_calls"],
            ]
            for m, v in ha["data"].items()
        ]
        out += [
            "",
            "## H-a retrieval (two-hop QA)",
            "",
            _table(["mode", "recall@1", "recall@5", "ms/query", "response bytes", "calls"], rows),
        ]
    if hb:
        steps = sorted(hb["data"]["sparse"], key=int)
        rows = [[arm, *[hb["data"][arm][s] for s in steps]] for arm in hb["data"]]
        out += [
            "",
            "## H-b precision on old associations after N new updates",
            "",
            _table(["arm", *[f"N={s}" for s in steps]], rows),
        ]
    if hb_e2e:
        cfg = hb_e2e["config"]
        rows = [
            [
                arm,
                v["taught"],
                v["retained"],
                v["retention"] if v["retention"] is not None else "-",
                v["taught@1"],
                v["retained@1"],
            ]
            for arm, v in hb_e2e["data"].items()
        ]
        out += [
            "",
            f"## H-b-e2e retrieval: recall@{cfg['k']} of the entity for alias questions "
            f"({cfg['n_set_a']} taught items, {cfg['n_interference']} interference updates, "
            f"{len(cfg['seeds'])} seeds)",
            "",
            _table(
                [
                    "arm",
                    "taught",
                    "retained (after B + interference)",
                    "retention",
                    "taught@1",
                    "retained@1",
                ],
                rows,
            ),
            "",
            "alias_ingest_clean and fly_update_consolidate are informational: the first is the "
            "alias baseline without the interference updates, the second adds one consolidate "
            "before the final test. (An earlier version of this note blamed unrelated updates "
            "for polluting every arm; that was a boost-normalisation bug in memory.recall, "
            "fixed with BOOST_SCALE.)",
        ]
    if hmcp:
        rows = [
            [
                k + (" (SIMULATED)" if v.get("simulated") else ""),
                v["tool_calls"],
                v["tool_list_bytes"],
                v["request_bytes"],
                v["response_bytes"],
                v["est_tokens"],
            ]
            for k, v in hmcp["data"].items()
        ]
        out += [
            "",
            "## H-mcp composed task (ingest, recall, update, recall)",
            "",
            _table(
                ["design", "calls", "tool list B", "request B", "response B", "est_tokens"], rows
            ),
        ]
    if llm and llm.get("status") == "OK":
        rows = [
            [
                h,
                arm,
                v["accuracy"],
                v["mean_prompt_tokens"],
                v["mean_completion_tokens"],
                v["mean_tool_calls"],
                v.get("reason_use_rate", "-"),
                v.get("values_accuracy", "-"),
            ]
            for h, arms in _llm_rows(llm)
            for arm, v in arms.items()
        ]
        out += [
            "",
            "## End-to-end with the live model",
            "",
            _table(
                [
                    "hypothesis",
                    "arm",
                    "accuracy",
                    "prompt tok/q",
                    "completion tok/q",
                    "tool calls/q",
                    "reason use",
                    "values acc (diagnostic, format-lenient)",
                ],
                rows,
            ),
        ]
        cal = calibration_rows(llm)
        if cal:
            out += [
                "",
                "## Calibration vs measured outcome",
                "",
                _table(
                    ["thinking", "family", "calibration score", "tool - alone", "sign agrees"], cal
                ),
                "",
                "Sign agreement: score >= 0.8 iff the tool helps. n is tiny (one calibration per "
                "mode, 17 hard items): no claim of statistical significance is made.",
            ]
    return "\n".join(out) + "\n\n" + MANUAL_MARKER + "\n"


def write(results_dir: Path = RESULTS_DIR, report_path: Path = REPORT_PATH) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manual = split_manual(report_path.read_text()) if report_path.exists() else ""
    report_path.write_text(render(results_dir) + manual)
    return report_path


if __name__ == "__main__":
    print(write())
