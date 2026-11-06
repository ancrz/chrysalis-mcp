"""Resource-guarded launcher for the local llama.cpp server (Gemma 4 26B-A4B on a Radeon 780M iGPU).

Why: an unguarded ``llama-server -ngl 99 -c 16384`` exhausted the iGPU GTT shared with the desktop
(amdgpu "Not enough memory for command submission" -> gnome-shell abort -> session restart).
cgroups cannot cap amdgpu GTT, so safety is: admission control before start, a reduced GPU
footprint (``--cpu-moe``, smaller context), RAM/CPU caps via a transient user unit, and a watchdog
unit that stops the server when GTT or RAM headroom falls below a floor.

CLI: ``python -m validation.llm_server preflight|start|stop|status|watchdog [--dry-run] [--force]``.
Every external effect (systemctl / systemd-run) goes through an injected ``Runner`` so tests never
touch systemd or the GPU. All subprocess calls use argv lists (no shell).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation import RESULTS_DIR, ROOT

MIB = 1024 * 1024
AMD_VENDOR_ID = "0x1002"
LLM_UNIT = "chrysalis-llm"
WATCHDOG_UNIT = "chrysalis-llm-watchdog"
WATCHDOG_LOG = RESULTS_DIR / "llm_watchdog.log"
HEALTH_TIMEOUT_S = 2.0

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    return int(env.get(key, default))


MEMORY_PROFILES = {"minimal": ("--no-host", "--no-repack"), "fast": ()}


@dataclass(frozen=True)
class Config:
    """All knobs; every value is overridable through ``CHRYSALIS_LLM_*`` env vars."""

    binary: Path
    model: Path
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 8192
    threads: int = 8
    alias: str = "gemma-4-26b-a4b"
    memory_max: str = "18G"
    memory_swap_max: str = "2G"
    cpu_quota: str = "1200%"
    nice: int = 10
    est_gpu_mib: int = 4096  # non-expert weights + KV(ctx) on the iGPU: conservative, never assumed
    est_ram_mib: int = 15360  # expert weights (mmap, page cache) + buffers
    gtt_margin_mib: int = 4096  # must remain free in GTT after the server's footprint
    ram_margin_mib: int = 6144  # must remain available in RAM after the server's footprint
    gtt_floor_mib: int = 2048  # watchdog: stop when GTT free drops below
    ram_floor_mib: int = 3072  # watchdog: stop when MemAvailable drops below
    watchdog_period_s: float = 2.0  # steady state
    watchdog_load_period_s: float = 0.5  # load phase: memory is pinned fastest right after start
    watchdog_load_window_s: float = 120.0
    memory_profile: str = "minimal"  # "minimal": --no-host --no-repack | "fast": neither
    sysfs: Path | None = None  # drm device dir; None = auto-detect
    meminfo: Path = Path("/proc/meminfo")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, profile: str | None = None) -> Config:
        """``profile`` (or CHRYSALIS_LLM_PROFILE) sets the defaults; explicit env vars still win."""
        e = dict(os.environ if env is None else env)
        home = Path(e.get("HOME", str(Path.home())))
        name = profile or e.get("CHRYSALIS_LLM_PROFILE") or DEFAULT_PROFILE
        if name not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {name!r}")
        prof = PROFILES[name]
        d = {
            k: type("D", (), {"default": prof.get(k, v.default)})
            for k, v in cls.__dataclass_fields__.items()
        }
        sysfs = e.get("CHRYSALIS_SYSFS_DRM")
        return cls(
            binary=Path(
                e.get(
                    "CHRYSALIS_LLM_BIN",
                    home / ".local/opt/llama.cpp/b12110/extract/llama-b12110/llama-server",
                )
            ),
            model=Path(
                e.get(
                    "CHRYSALIS_LLM_MODEL",
                    home / MODEL_DIR / prof.get("model", "gemma-4-26B-A4B-it-Q4_0.gguf"),
                )
            ),
            port=_env_int(e, "CHRYSALIS_LLM_PORT", d["port"].default),
            ctx=_env_int(e, "CHRYSALIS_LLM_CTX", d["ctx"].default),
            threads=_env_int(e, "CHRYSALIS_LLM_THREADS", d["threads"].default),
            est_gpu_mib=_env_int(e, "CHRYSALIS_LLM_EST_GPU_MIB", d["est_gpu_mib"].default),
            est_ram_mib=_env_int(e, "CHRYSALIS_LLM_EST_RAM_MIB", d["est_ram_mib"].default),
            gtt_margin_mib=_env_int(e, "CHRYSALIS_LLM_GTT_MARGIN_MIB", d["gtt_margin_mib"].default),
            ram_margin_mib=_env_int(e, "CHRYSALIS_LLM_RAM_MARGIN_MIB", d["ram_margin_mib"].default),
            gtt_floor_mib=_env_int(e, "CHRYSALIS_LLM_GTT_FLOOR_MIB", d["gtt_floor_mib"].default),
            ram_floor_mib=_env_int(e, "CHRYSALIS_LLM_RAM_FLOOR_MIB", d["ram_floor_mib"].default),
            alias=e.get("CHRYSALIS_LLM_ALIAS", d["alias"].default),
            sysfs=Path(sysfs) if sysfs else None,
            memory_profile=_profile(e.get("CHRYSALIS_LLM_MEMORY_PROFILE", "minimal")),
        )


PROFILES: dict[str, dict[str, Any]] = {
    "26b": {},  # the Config defaults (Gemma 4 26B-A4B)
    "e4b": {
        "model": "gemma-4-E4B-it-Q4_0.gguf",
        "alias": "gemma-4-e4b",
        "ctx": 8192,
        "est_gpu_mib": 6144,
        "est_ram_mib": 2048,
        "threads": 6,
        "ram_margin_mib": 4096,
    },
}
DEFAULT_PROFILE = "26b"
MODEL_DIR = ".local/share/chrysalis/models"


def _profile(name: str) -> str:
    if name not in MEMORY_PROFILES:
        raise ValueError(f"CHRYSALIS_LLM_MEMORY_PROFILE must be one of {sorted(MEMORY_PROFILES)}")
    return name


@dataclass(frozen=True)
class Resources:
    """Free memory in MiB; ``gtt_free_mib`` is None when the GPU counters are unreadable."""

    gtt_free_mib: int | None
    vram_free_mib: int | None
    ram_available_mib: int | None
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    admitted: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------------------- probes
def detect_sysfs(root: Path = Path("/sys/class/drm")) -> Path | None:
    """First ``card*/device`` whose vendor is AMD and which exposes the GTT counters."""
    for card in sorted(root.glob("card[0-9]*")):
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor == AMD_VENDOR_ID and (dev / "mem_info_gtt_total").exists():
            return dev
    return None


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def read_resources(cfg: Config) -> Resources:
    problems: list[str] = []
    gtt_free = vram_free = ram_avail = None
    sysfs = cfg.sysfs or detect_sysfs()
    if sysfs is None:
        problems.append("no AMD drm device with GTT counters found (sysfs unreadable)")
    else:
        try:
            gtt_free = (
                _read_int(sysfs / "mem_info_gtt_total") - _read_int(sysfs / "mem_info_gtt_used")
            ) // MIB
            vram_free = (
                _read_int(sysfs / "mem_info_vram_total") - _read_int(sysfs / "mem_info_vram_used")
            ) // MIB
        except (OSError, ValueError) as exc:
            problems.append(f"cannot read GPU memory counters in {sysfs}: {exc}")
    try:
        for line in cfg.meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                ram_avail = int(line.split()[1]) // 1024  # kB -> MiB
                break
        else:
            problems.append("MemAvailable missing from meminfo")
    except (OSError, ValueError) as exc:
        problems.append(f"cannot read meminfo {cfg.meminfo}: {exc}")
    return Resources(gtt_free, vram_free, ram_avail, tuple(problems))


# ----------------------------------------------------------------------------------- decisions
def admit(res: Resources, cfg: Config) -> Decision:
    """Admit only if, after the server's estimated footprint, both margins still hold."""
    reasons = list(res.problems)
    if res.gtt_free_mib is not None and res.gtt_free_mib - cfg.est_gpu_mib < cfg.gtt_margin_mib:
        reasons.append(
            f"GTT free {res.gtt_free_mib} MiB - est {cfg.est_gpu_mib} < margin {cfg.gtt_margin_mib}"
        )
    if res.ram_available_mib is not None and (
        res.ram_available_mib - cfg.est_ram_mib < cfg.ram_margin_mib
    ):
        reasons.append(
            f"MemAvailable {res.ram_available_mib} MiB - est {cfg.est_ram_mib} "
            f"< margin {cfg.ram_margin_mib}"
        )
    return Decision(not reasons, tuple(reasons))


def watchdog_decision(res: Resources, cfg: Config) -> str | None:
    """Reason to stop the server now, or None. Unreadable counters count as a reason (fail safe)."""
    if res.gtt_free_mib is None or res.ram_available_mib is None:
        return "resource counters unreadable: " + "; ".join(res.problems)
    if res.gtt_free_mib < cfg.gtt_floor_mib:
        return f"GTT free {res.gtt_free_mib} MiB < floor {cfg.gtt_floor_mib}"
    if res.ram_available_mib < cfg.ram_floor_mib:
        return f"MemAvailable {res.ram_available_mib} MiB < floor {cfg.ram_floor_mib}"
    return None


# ----------------------------------------------------------------------------------- argv builders
def server_argv(cfg: Config) -> list[str]:
    """``systemd-run`` argv starting llama-server as a capped transient user service."""
    return [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={LLM_UNIT}",
        "-p",
        f"MemoryMax={cfg.memory_max}",
        "-p",
        f"MemorySwapMax={cfg.memory_swap_max}",
        "-p",
        f"CPUQuota={cfg.cpu_quota}",
        "-p",
        f"Nice={cfg.nice}",
        str(cfg.binary),
        "-m",
        str(cfg.model),
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--cpu-moe",
        "-ngl",
        "99",
        "-c",
        str(cfg.ctx),
        "--fit",
        "on",
        "--fit-target",
        str(cfg.gtt_margin_mib),
        "-t",
        str(cfg.threads),
        "--jinja",
        *MEMORY_PROFILES[cfg.memory_profile],
        "--alias",
        cfg.alias,
    ]


def watchdog_argv(cfg: Config, env: dict[str, str] | None = None) -> list[str]:
    """``systemd-run`` argv for the watchdog unit, bound to the server unit's lifetime.

    Transient units do not inherit the caller's environment, so every ``CHRYSALIS_*`` variable is
    forwarded explicitly (the watchdog must read the same sysfs/thresholds as the preflight).
    """
    source = os.environ if env is None else env
    forwarded: list[str] = []
    for key in sorted(k for k in source if k.startswith("CHRYSALIS_")):
        forwarded += ["-p", f'Environment="{key}={source[key]}"']
    return [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={WATCHDOG_UNIT}",
        "-p",
        f"BindsTo={LLM_UNIT}.service",
        "-p",
        f"After={LLM_UNIT}.service",
        "-p",
        f"WorkingDirectory={ROOT}",
        "-p",
        "Nice=5",
        *forwarded,
        sys.executable,
        "-m",
        "validation.llm_server",
        "watchdog",
    ]


def _stop_argv(unit: str) -> list[str]:
    return ["systemctl", "--user", "stop", f"{unit}.service"]


# --------------------------------------------------------------------------------------- actions
def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=30)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def poll_interval(cfg: Config, elapsed_s: float) -> float:
    """Fast polling during the load phase after start, steady-state afterwards."""
    return (
        cfg.watchdog_load_period_s
        if elapsed_s < cfg.watchdog_load_window_s
        else cfg.watchdog_period_s
    )


def _try(runner: Runner, argv: Sequence[str], log_path: Path) -> bool:
    """Run one control command; any failure (non-zero, timeout, other error) is logged -> False."""
    try:
        res = runner(argv)
    except Exception as exc:  # noqa: BLE001 - the safety path must never raise
        _log(log_path, f"{' '.join(argv)} raised {type(exc).__name__}: {exc}")
        return False
    if res.returncode != 0:
        _log(log_path, f"{' '.join(argv)} exited {res.returncode}: {(res.stderr or '').strip()}")
    return res.returncode == 0


def stop_server(
    runner: Runner = _run,
    log_path: Path = WATCHDOG_LOG,
    kill: Callable[[int, int], None] = os.kill,
) -> bool:
    """Escalating stop: stop, stop again, systemctl SIGKILL, then SIGKILL of the MainPID."""
    stop = _stop_argv(LLM_UNIT)
    if _try(runner, stop, log_path) or _try(runner, stop, log_path):
        return True
    if _try(
        runner, ["systemctl", "--user", "kill", "--signal=SIGKILL", f"{LLM_UNIT}.service"], log_path
    ):
        return True
    try:
        shown = runner(
            ["systemctl", "--user", "show", "-p", "MainPID", "--value", f"{LLM_UNIT}.service"]
        )
        pid = int(shown.stdout.strip() or 0)
        if pid > 1:
            kill(pid, signal.SIGKILL)
            _log(log_path, f"last resort: SIGKILL sent to MainPID {pid}")
            return True
        _log(log_path, f"no killable MainPID ({pid}); unit probably already gone")
    except Exception as exc:  # noqa: BLE001
        _log(log_path, f"last-resort kill failed: {type(exc).__name__}: {exc}")
    return False


def watchdog_loop(
    cfg: Config,
    runner: Runner = _run,
    sleep: Callable[[float], None] = time.sleep,
    log_path: Path = WATCHDOG_LOG,
    max_iters: int | None = None,
    kill: Callable[[int, int], None] = os.kill,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """Poll resources; stop the server on a breach. Returns the reason once the server is down.

    Never dies on its own errors: an exception or a failed stop is logged and the next tick retries
    (the unit is BindsTo the server, so it ends when the server does).
    """
    iters = 0
    t_start = clock()
    while max_iters is None or iters < max_iters:
        iters += 1
        try:
            reason = watchdog_decision(read_resources(cfg), cfg)
            if reason:
                _log(log_path, f"STOP {LLM_UNIT}: {reason}")
                if stop_server(runner, log_path, kill):
                    return reason
        except Exception as exc:  # noqa: BLE001
            with suppress(OSError):
                _log(log_path, f"watchdog tick failed: {type(exc).__name__}: {exc}")
        sleep(poll_interval(cfg, clock() - t_start))
    return None


def health(cfg: Config) -> str:
    conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=HEALTH_TIMEOUT_S)
    try:
        conn.request("GET", "/health")
        return f"HTTP {conn.getresponse().status}"
    except (OSError, http.client.HTTPException) as exc:
        return f"unreachable ({type(exc).__name__})"
    finally:
        conn.close()


def _state(res: Resources) -> dict[str, Any]:
    return {
        "gtt_free_mib": res.gtt_free_mib,
        "vram_free_mib": res.vram_free_mib,
        "ram_available_mib": res.ram_available_mib,
        "problems": list(res.problems),
    }


def cmd_preflight(cfg: Config, out: Callable[[str], None]) -> int:
    res = read_resources(cfg)
    dec = admit(res, cfg)
    out(json.dumps({"admitted": dec.admitted, "reasons": list(dec.reasons), **_state(res)}))
    return 0 if dec.admitted else 2


def cmd_start(
    cfg: Config, *, dry_run: bool, force: bool, runner: Runner, out: Callable[[str], None]
) -> int:
    res = read_resources(cfg)
    dec = admit(res, cfg)
    if dry_run:
        out(
            json.dumps(
                {
                    "dry_run": True,
                    "admitted": dec.admitted,
                    "reasons": list(dec.reasons),
                    "server_argv": server_argv(cfg),
                    "watchdog_argv": watchdog_argv(cfg),
                }
            )
        )
        return 0
    if not dec.admitted:
        out(f"REFUSED: {'; '.join(dec.reasons)}")
        if not force:
            return 2
        out("RISK: --force given; starting despite refusal (can crash the desktop session)")
    for path in (cfg.binary, cfg.model):
        if not path.exists():
            out(f"REFUSED: missing file {path}")
            return 2
    started = runner(server_argv(cfg))
    if started.returncode != 0:
        err = started.stderr.strip()
        if "already" in err.lower():
            out(f"REFUSED: unit {LLM_UNIT} already exists (running?). Run `stop` first. {err}")
        else:
            out(f"systemd-run failed: {err}")
        return 1
    guard = runner(watchdog_argv(cfg))
    if guard.returncode != 0:  # never leave the server running unguarded
        runner(_stop_argv(LLM_UNIT))
        out(f"watchdog failed to start, server stopped: {guard.stderr.strip()}")
        return 1
    out(f"started {LLM_UNIT} + {WATCHDOG_UNIT} on {cfg.host}:{cfg.port}")
    return 0


def cmd_stop(runner: Runner, out: Callable[[str], None]) -> int:
    rcs = [runner(_stop_argv(u)).returncode for u in (WATCHDOG_UNIT, LLM_UNIT)]
    out(f"stop issued (exit codes watchdog/server: {rcs})")
    return 0


def _loaded_model(runner: Runner) -> str | None:
    """Model filename from the running unit's argv (``-m <path>``), None when not running."""
    res = runner(
        ["systemctl", "--user", "show", "-p", "ExecStart", "--value", f"{LLM_UNIT}.service"]
    )
    match = re.search(r"argv\[\]=\S*llama-server\s.*?-m\s+(\S+)", res.stdout or "")
    return Path(match.group(1)).name if match else None


def cmd_status(
    cfg: Config,
    runner: Runner,
    out: Callable[[str], None],
    probe: Callable[[Config], str] = health,
) -> int:
    active = {
        u: runner(["systemctl", "--user", "is-active", f"{u}.service"]).stdout.strip() or "unknown"
        for u in (LLM_UNIT, WATCHDOG_UNIT)
    }
    model = _loaded_model(runner) or f"{cfg.model.name} (config, not running)"
    out(
        json.dumps(
            {"units": active, "model": model, **_state(read_resources(cfg)), "health": probe(cfg)}
        )
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = _run,
    cfg: Config | None = None,
    out: Callable[[str], None] = print,
) -> int:
    ap = argparse.ArgumentParser(prog="python -m validation.llm_server")
    ap.add_argument("command", choices=["preflight", "start", "stop", "status", "watchdog"])
    ap.add_argument("--dry-run", action="store_true", help="start: print argv, run nothing")
    ap.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="model preset (env CHRYSALIS_LLM_PROFILE; default 26b). The lab uses e4b",
    )
    ap.add_argument("--force", action="store_true", help="start despite a refused admission")
    args = ap.parse_args(argv)
    cfg = cfg or Config.from_env(profile=args.profile)
    if args.command == "preflight":
        return cmd_preflight(cfg, out)
    if args.command == "start":
        return cmd_start(cfg, dry_run=args.dry_run, force=args.force, runner=runner, out=out)
    if args.command == "stop":
        return cmd_stop(runner, out)
    if args.command == "status":
        return cmd_status(cfg, runner, out)
    watchdog_loop(cfg, runner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
