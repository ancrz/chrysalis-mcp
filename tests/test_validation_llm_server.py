"""llm_server guardrails: fake sysfs/meminfo + injected runner. Never touches systemd or the GPU."""

import json
import subprocess
from pathlib import Path

import pytest

from validation import llm_server as ls
from validation.llm_server import MIB, Config, Resources

GIB = 1024


def make_cfg(tmp_path: Path, gtt_free_mib=12000, ram_avail_mib=30000, sysfs=True, **kw) -> Config:
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    if sysfs:
        total = 15000 * MIB
        (dev / "mem_info_gtt_total").write_text(str(total))
        (dev / "mem_info_gtt_used").write_text(str(total - gtt_free_mib * MIB))
        (dev / "mem_info_vram_total").write_text(str(1024 * MIB))
        (dev / "mem_info_vram_used").write_text(str(100 * MIB))
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: 99999999 kB\nMemAvailable: {ram_avail_mib * 1024} kB\n")
    return Config(
        binary=tmp_path / "llama-server",
        model=tmp_path / "m.gguf",
        sysfs=dev if sysfs else tmp_path / "absent",
        meminfo=meminfo,
        **kw,
    )


class FakeRunner:
    def __init__(self, rcs=None, stdout=""):
        self.calls: list[list[str]] = []
        self.rcs = rcs or {}
        self.stdout = stdout

    def __call__(self, argv):
        self.calls.append(list(argv))
        rc = self.rcs.get(argv[0 if argv[0] != "systemctl" else 2], 0)
        return subprocess.CompletedProcess(argv, rc, self.stdout, "boom")


# est_gpu 4096 + margin 4096 = 8192 needed in GTT; est_ram 15360 + margin 6144 = 21504 in RAM
@pytest.mark.parametrize(
    "gtt, ram, ok",
    [
        (12000, 30000, True),
        (8192, 30000, True),  # exact margin admits
        (8191, 30000, False),  # one MiB under
        (12000, 21504, True),
        (12000, 21503, False),
        (3000, 5000, False),
    ],
)
def test_admission_table(tmp_path, gtt, ram, ok):
    cfg = make_cfg(tmp_path, gtt, ram)
    dec = ls.admit(ls.read_resources(cfg), cfg)
    assert dec.admitted is ok
    assert bool(dec.reasons) is (not ok)


def test_missing_sysfs_refuses_with_a_reason(tmp_path):
    cfg = make_cfg(tmp_path, sysfs=False)
    dec = ls.admit(ls.read_resources(cfg), cfg)
    assert not dec.admitted
    assert "GPU" in " ".join(dec.reasons)


def test_detect_sysfs_requires_amd_vendor_and_gtt(tmp_path):
    amd = tmp_path / "card0" / "device"
    other = tmp_path / "card1" / "device"
    for d, vendor in ((amd, "0x1002"), (other, "0x8086")):
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "mem_info_gtt_total").write_text("1")
    assert ls.detect_sysfs(tmp_path) == amd
    assert ls.detect_sysfs(tmp_path / "nowhere") is None


@pytest.mark.parametrize(
    "gtt, ram, stop",
    [
        (2048, 3072, False),  # at the floors: keep running
        (2047, 9000, True),
        (9000, 3071, True),
        (None, 9000, True),  # unreadable counters fail safe
    ],
)
def test_watchdog_decision_table(tmp_path, gtt, ram, stop):
    cfg = make_cfg(tmp_path)
    res = Resources(gtt, 0, ram, () if gtt is not None else ("gtt unreadable",))
    assert (ls.watchdog_decision(res, cfg) is not None) is stop


def test_watchdog_loop_stops_the_server_and_logs(tmp_path):
    cfg = make_cfg(tmp_path, gtt_free_mib=1000)
    runner, log = FakeRunner(), tmp_path / "wd.log"
    reason = ls.watchdog_loop(cfg, runner, sleep=lambda s: None, log_path=log, max_iters=3)
    assert "GTT free 1000" in reason
    assert runner.calls == [["systemctl", "--user", "stop", "chrysalis-llm.service"]]
    assert "STOP chrysalis-llm" in log.read_text()


def test_watchdog_loop_is_quiet_when_healthy(tmp_path):
    cfg = make_cfg(tmp_path)
    runner, sleeps = FakeRunner(), []
    assert ls.watchdog_loop(cfg, runner, sleeps.append, tmp_path / "l", max_iters=3) is None
    assert runner.calls == [] and sleeps == [cfg.watchdog_load_period_s] * 3


def test_dry_run_emits_exact_capped_argv_and_runs_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    out, runner = [], FakeRunner()
    assert ls.main(["start", "--dry-run"], runner=runner, cfg=cfg, out=out.append) == 0
    assert runner.calls == []
    payload = json.loads(out[0])
    argv = payload["server_argv"]
    assert argv[:4] == ["systemd-run", "--user", "--collect", "--unit=chrysalis-llm"]
    for prop in ("MemoryMax=18G", "MemorySwapMax=2G", "CPUQuota=1200%", "Nice=10"):
        assert argv[argv.index(prop) - 1] == "-p"
    assert argv[argv.index(str(cfg.binary)) :] == [
        str(cfg.binary), "-m", str(cfg.model), "--host", "127.0.0.1", "--port", "8080",
        "--cpu-moe", "-ngl", "99", "-c", "8192", "--fit", "on", "--fit-target", "4096",
        "-t", "8", "--jinja", "--no-host", "--no-repack", "--alias", "gemma-4-26b-a4b",
    ]  # fmt: skip
    wd = payload["watchdog_argv"]
    assert "BindsTo=chrysalis-llm.service" in wd and wd[-2:] == [
        "validation.llm_server",
        "watchdog",
    ]


def test_start_refuses_without_admission_and_force_still_warns(tmp_path):
    cfg = make_cfg(tmp_path, gtt_free_mib=100)
    cfg.binary.write_text("")
    cfg.model.write_text("")
    out, runner = [], FakeRunner()
    assert ls.main(["start"], runner=runner, cfg=cfg, out=out.append) == 2
    assert runner.calls == [] and out[0].startswith("REFUSED")
    out.clear()
    assert ls.main(["start", "--force"], runner=runner, cfg=cfg, out=out.append) == 0
    assert any("RISK" in line for line in out)
    assert [c[0] for c in runner.calls] == ["systemd-run", "systemd-run"]


def test_start_refuses_when_files_are_missing(tmp_path):
    cfg = make_cfg(tmp_path)
    runner = FakeRunner()
    assert ls.main(["start"], runner=runner, cfg=cfg, out=lambda s: None) == 2
    assert runner.calls == []


def test_server_is_stopped_if_the_watchdog_cannot_start(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.binary.write_text("")
    cfg.model.write_text("")

    class R(FakeRunner):
        def __call__(self, argv):
            res = super().__call__(argv)
            if "--unit=chrysalis-llm-watchdog" in argv:
                res.returncode = 1
            return res

    runner = R()
    assert ls.main(["start"], runner=runner, cfg=cfg, out=lambda s: None) == 1
    assert runner.calls[-1] == ["systemctl", "--user", "stop", "chrysalis-llm.service"]


def test_preflight_status_and_stop_use_only_the_injected_runner(tmp_path):
    cfg = make_cfg(tmp_path)
    out, runner = [], FakeRunner(stdout="active\n")
    assert ls.main(["preflight"], runner=runner, cfg=cfg, out=out.append) == 0
    assert json.loads(out[0])["admitted"] is True
    out.clear()
    assert ls.cmd_status(cfg, runner, out.append, probe=lambda c: "HTTP 200") == 0
    assert json.loads(out[0])["units"] == {
        "chrysalis-llm": "active",
        "chrysalis-llm-watchdog": "active",
    }
    assert ls.main(["stop"], runner=runner, cfg=cfg, out=out.append) == 0
    assert all(c[0] == "systemctl" for c in runner.calls)


def test_config_from_env_overrides(tmp_path):
    cfg = Config.from_env({"HOME": "/h", "CHRYSALIS_LLM_CTX": "4096", "CHRYSALIS_SYSFS_DRM": "/x"})
    assert cfg.ctx == 4096 and cfg.sysfs == Path("/x")
    assert str(cfg.model).startswith("/h/.local/share/chrysalis/models/")


STOP = ["systemctl", "--user", "stop", "chrysalis-llm.service"]
KILL = ["systemctl", "--user", "kill", "--signal=SIGKILL", "chrysalis-llm.service"]
SHOW = ["systemctl", "--user", "show", "-p", "MainPID", "--value", "chrysalis-llm.service"]


class Script:
    """Runner answering per argv: int = return code, Exception = raised, str = stdout (rc 0)."""

    def __init__(self, answers):
        self.answers = {tuple(k): list(v) for k, v in answers.items()}
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        queue = self.answers.get(tuple(argv), [0])
        ans = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(ans, Exception):
            raise ans
        rc, out = (ans, "") if isinstance(ans, int) else (0, ans)
        return subprocess.CompletedProcess(argv, rc, out, "err")


def test_stop_failure_is_retried_once_then_succeeds(tmp_path):
    r = Script({tuple(STOP): [1, 0]})
    assert ls.stop_server(r, tmp_path / "l") is True
    assert r.calls == [STOP, STOP]


def test_stop_escalates_stop_stop_kill_then_mainpid(tmp_path):
    killed = []
    r = Script({tuple(STOP): [1], tuple(KILL): [1], tuple(SHOW): ["4242\n"]})
    assert ls.stop_server(r, tmp_path / "l", lambda pid, sig: killed.append((pid, sig))) is True
    assert r.calls == [STOP, STOP, KILL, SHOW]
    assert killed == [(4242, ls.signal.SIGKILL)]


def test_stop_timeouts_escalate_the_same_way(tmp_path):
    timeout = subprocess.TimeoutExpired("systemctl", 30)
    r = Script({tuple(STOP): [timeout], tuple(KILL): [0]})
    assert ls.stop_server(r, tmp_path / "l") is True
    assert r.calls == [STOP, STOP, KILL]


def test_stop_server_reports_failure_without_raising_when_everything_fails(tmp_path):
    r = Script({tuple(STOP): [1], tuple(KILL): [RuntimeError("x")], tuple(SHOW): ["0"]})
    assert (
        ls.stop_server(r, tmp_path / "l", lambda p, s: pytest.fail("must not kill pid 0")) is False
    )
    assert "no killable MainPID" in (tmp_path / "l").read_text()


def test_loop_survives_an_exception_and_a_failed_stop_then_stops(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, gtt_free_mib=1000)  # breach on every tick
    real, n = ls.read_resources, {"i": 0}

    def flaky(c):
        n["i"] += 1
        if n["i"] == 1:
            raise OSError("sysfs vanished")
        return real(c)

    monkeypatch.setattr(ls, "read_resources", flaky)
    r = Script({tuple(STOP): [1, 1, 1, 1, 0]})  # tick 2: stop x2 fail, kill ok->True
    sleeps = []
    log = tmp_path / "wd.log"
    reason = ls.watchdog_loop(cfg, r, sleeps.append, log, max_iters=5, kill=lambda p, s: None)
    assert reason is not None and len(sleeps) == 1  # slept only after the exception tick
    assert "tick failed: OSError" in log.read_text()
    assert r.calls[:3] == [STOP, STOP, KILL]


def test_loop_keeps_running_after_every_stop_path_fails(tmp_path):
    cfg = make_cfg(tmp_path, gtt_free_mib=1000)
    r = Script({tuple(STOP): [1], tuple(KILL): [1], tuple(SHOW): ["0"]})
    sleeps = []
    assert ls.watchdog_loop(cfg, r, sleeps.append, tmp_path / "l", max_iters=3) is None
    assert len(sleeps) == 3 and r.calls.count(STOP) == 6


def test_watchdog_unit_receives_the_chrysalis_environment(tmp_path):
    cfg = make_cfg(tmp_path)
    env = {"CHRYSALIS_SYSFS_DRM": "/x y", "CHRYSALIS_LLM_CTX": "4096", "HOME": "/h"}
    argv = ls.watchdog_argv(cfg, env)
    assert argv[argv.index('Environment="CHRYSALIS_LLM_CTX=4096"') - 1] == "-p"
    assert 'Environment="CHRYSALIS_SYSFS_DRM=/x y"' in argv
    assert not any("HOME" in a for a in argv)
    assert argv.index("Nice=5") < argv.index('Environment="CHRYSALIS_LLM_CTX=4096"')


def test_both_units_are_collected(tmp_path):
    cfg = make_cfg(tmp_path)
    assert "--collect" in ls.server_argv(cfg) and "--collect" in ls.watchdog_argv(cfg, {})


def test_duplicate_start_reports_clearly_and_starts_no_watchdog(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.binary.write_text("")
    cfg.model.write_text("")

    class Dup(FakeRunner):
        def __call__(self, argv):
            res = super().__call__(argv)
            res.returncode, res.stderr = 1, "Failed: Unit chrysalis-llm.service already exists."
            return res

    out, r = [], Dup()
    assert ls.main(["start"], runner=r, cfg=cfg, out=out.append) == 1
    assert [c[0] for c in r.calls] == ["systemd-run"]  # no watchdog, no stop
    assert "already exists" in out[0] and "stop" in out[0]


@pytest.mark.parametrize(
    "profile, flags",
    [("minimal", ["--no-host", "--no-repack"]), ("fast", [])],
)
def test_memory_profile_selects_flags(tmp_path, profile, flags):
    argv = ls.server_argv(make_cfg(tmp_path, memory_profile=profile))
    assert [f for f in ("--no-host", "--no-repack") if f in argv] == flags
    assert "--mmap" not in argv  # llama.cpp default; the binary is never probed (GPU safety)


def test_memory_profile_env_default_and_validation():
    assert Config.from_env({"HOME": "/h"}).memory_profile == "minimal"
    assert (
        Config.from_env({"HOME": "/h", "CHRYSALIS_LLM_MEMORY_PROFILE": "fast"}).memory_profile
        == "fast"
    )
    with pytest.raises(ValueError):
        Config.from_env({"HOME": "/h", "CHRYSALIS_LLM_MEMORY_PROFILE": "turbo"})


def test_poll_interval_is_fast_during_load_then_steady(tmp_path):
    cfg = make_cfg(tmp_path)
    assert [ls.poll_interval(cfg, t) for t in (0, 119.9, 120, 500)] == [0.5, 0.5, 2.0, 2.0]


def test_loop_sleeps_follow_the_schedule_with_an_injected_clock(tmp_path):
    cfg = make_cfg(tmp_path)
    times = iter([0.0, 10.0, 119.0, 121.0, 300.0])  # t_start, then one reading per sleep
    sleeps = []
    assert (
        ls.watchdog_loop(
            cfg, FakeRunner(), sleeps.append, tmp_path / "l", 4, clock=lambda: next(times)
        )
        is None
    )
    assert sleeps == [0.5, 0.5, 2.0, 2.0]


# ------------------------------------------------------------------------------------ profiles
def test_profile_e4b_sets_the_six_fields_and_env_still_overrides():
    cfg = Config.from_env({"HOME": "/h"}, profile="e4b")
    assert cfg.model == Path("/h/.local/share/chrysalis/models/gemma-4-E4B-it-Q4_0.gguf")
    assert (cfg.ctx, cfg.est_gpu_mib, cfg.est_ram_mib, cfg.threads, cfg.ram_margin_mib) == (
        8192,
        6144,
        2048,
        6,
        4096,
    )
    over = Config.from_env(
        {"HOME": "/h", "CHRYSALIS_LLM_PROFILE": "e4b", "CHRYSALIS_LLM_CTX": "4096"}
    )
    assert over.ctx == 4096 and over.threads == 6 and "E4B" in over.model.name


def test_default_profile_is_26b_and_unknown_profile_is_rejected():
    cfg = Config.from_env({"HOME": "/h"})
    assert "26B-A4B" in cfg.model.name and cfg.est_ram_mib == 15360 and cfg.ram_margin_mib == 6144
    with pytest.raises(ValueError):
        Config.from_env({"HOME": "/h", "CHRYSALIS_LLM_PROFILE": "nope"})


def test_cli_profile_dry_run_builds_e4b_argv(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CHRYSALIS_LLM_MODEL", raising=False)
    out: list[str] = []
    ls.main(["start", "--dry-run", "--profile", "e4b"], runner=FakeRunner(), out=out.append)
    assert "gemma-4-E4B-it-Q4_0.gguf" in out[0]


def test_status_shows_the_loaded_model_from_unit_argv_or_config(tmp_path):
    cfg = make_cfg(tmp_path)
    argv = "{ argv[]=/x/llama-server -m /m/gemma-4-E4B-it-Q4_0.gguf --host 1 ; }"
    out: list[str] = []
    ls.cmd_status(cfg, FakeRunner(stdout=argv), out.append, probe=lambda c: "HTTP 200")
    assert json.loads(out[0])["model"] == "gemma-4-E4B-it-Q4_0.gguf"
    out.clear()
    ls.cmd_status(cfg, FakeRunner(stdout="inactive\n"), out.append, probe=lambda c: "down")
    assert json.loads(out[0])["model"] == "m.gguf (config, not running)"


def test_e4b_alias_and_env_override():
    assert Config.from_env({"HOME": "/h"}, profile="e4b").alias == "gemma-4-e4b"
    assert Config.from_env({"HOME": "/h"}).alias == "gemma-4-26b-a4b"
    env = {"HOME": "/h", "CHRYSALIS_LLM_ALIAS": "mine"}
    assert Config.from_env(env, profile="e4b").alias == "mine"
