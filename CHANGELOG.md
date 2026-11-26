# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-26

### Added

- Memory substrate v5 bio layer: negative selection (order-independent per-KC document frequency),
  quorum sensing (`learned_status` provisional or confirmed; `quorum` and `learned_min_status`
  recall params), opt-in replay and usage weighting in `consolidate`, and bounded edge-usage
  tracking in `recall`. Forward migration v4 to v5 included.
- Memory substrate v6 low-priority bio tools, all opt-in: tagging and capture in `consolidate`,
  adaptive inhibition, sparse distributed memory (`mode=sdm`), a FlyVec-lite encoder and
  `memory.flyvec_train`. Defaults are unchanged. Covered by the new `bench_bio_low` bench.
- Advisory learning: `meta.feedback`, `meta.advice_stats` and `meta.advice_reset`. Advice is learned
  per model from outcomes (Beta counts in a separate `advice.db`, `CHRYSALIS_ADVICE_DB`), returns
  the rule prior until `min_n` outcomes, then the better arm, with an evidence object.
- `meta.batch` resolves `$prev` and `$steps.<id>` references between steps.
- `memory.teach` as an alias of `memory.update`.
- `CHRYSALIS_BIO_DEFAULTS=all` server-wide switch: fills omitted bio params in `recall` and
  `consolidate` and retrains FlyVec after ingest (capped at 5000 nodes). Explicit params win.
  `bio_defaults` is reported in the schema, handshake and discover.
- Eval feedback loop (`CHRYSALIS_EVAL_FEEDBACK=1`): the advised arm reports each outcome through
  `meta.feedback`; each run gets its own advice db under the results directory.
- Tool-resistance arms on H-adv and H-adv-heldout: `fewshot`, `forced` (`tool_choice=required` on
  the first step) and `oracle_ctx` (labeled as an upper bound, not a fair arm).
- H-bio hypothesis: the bio actions scored per step, each step isolated and repeated.
- Eval throughput: `--concurrency`, lenient QA scoring, `finish_reason` and `truncated_rate` per
  arm, and per-mode token and timeout budgets.
- `.env.example` documenting every `CHRYSALIS_*` variable, with a test that guards drift in both
  directions; `.gitattributes` with eol and binary rules.
- GitHub Actions CI (ruff lint, format check and the default pytest selection).

### Changed

- Small-model robustness: params accepted as a JSON string or list (coerced and noted in
  `coerced`), param errors carry `did_you_mean` and the action example, top-level task features are
  nested into `task`, ingest triples are normalized, a Python-literal params fallback (64 KiB cap,
  literals only), curated auto-renames such as `used_tool` to `used`, and guarded JSON decoding.
- An implicit handshake happens when a call carries `model` (an explicit profile is kept).
- Feedback replies that are not JSON no longer abort an evaluation arm.
- The validation focus is Gemma 4 E4B (Q4_K_M) with reasoning on. The eval CLI defaults to
  thinking on, memory groups also run with it, and the context reserve follows the mode's answer
  budget. Published results were regenerated for that configuration.
- Brief schema byte budget raised from 8000 to 8500 for the new actions and params.
- Per-model raw validation outputs stay local; only summary JSON and `RESULTS.md` are published.
- Test files named after internal task numbers were renamed to descriptive names.

### Removed

- The Qwen and DeepSeek launcher profiles and their parameter entries; the default launcher
  profile is now `e4b-km`.
- The thinking-off published results and the off-mode default in the eval CLI.

### Fixed

- `AdviceStore` rows and reset use fully static SQL (no formatted WHERE clause).
- Capture SQL is fully static and parameterized.
- `meta.feedback` guards against advice-store lock errors.

## [0.1.0] - 2026-11-06

Initial release: single-tool MCP server with graph memory, continual learning outside the weights,
Z3-backed reasoning and evidence-based advice.

[0.2.0]: https://github.com/ancrz/chrysalis-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ancrz/chrysalis-mcp/releases/tag/v0.1.0
