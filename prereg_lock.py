"""Mechanical pre-registration lock (§0.5).

``analyze.py --exp e`` and ``--exp h`` refuse to run unless:

1. ``prereg.md`` exists **in git history**, with a commit timestamp earlier
   than the creation time of the Experiment E prediction files;
2. the working-tree copy is **byte-identical** to the committed version.

There is deliberately no flag to skip this.  ``check()`` raises
``PreregViolation``; nothing in this module accepts an override argument, so a
caller cannot pass one, and a future edit that adds one is a visible change to
this file rather than a command-line convenience.

§3.7 additionally requires the commit to be **pushed to a public repository** --
a local commit is rewritable and is weak pre-registration.  Push state cannot be
verified offline, so ``check()`` reports it as a warning with the measured
evidence (upstream ref, whether the commit is an ancestor of it) rather than
pretending to have verified it.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import config as C

PREREG_PATH = C.REPO_ROOT / "prereg.md"

#: Experiments whose analysis is sealed behind the lock.
LOCKED_EXPERIMENTS = ("e", "h")


class PreregViolation(RuntimeError):
    """Raised when the pre-registration lock fails.  Not catchable by a flag."""


@dataclass
class LockReport:
    passed: bool
    prereg_path: str
    exists_in_worktree: bool
    committed: bool
    commit_sha: str | None
    commit_unix: float | None
    commit_iso: str | None
    worktree_sha256: str | None
    committed_sha256: str | None
    identical: bool
    artifacts_checked: list
    earliest_artifact_unix: float | None
    commit_precedes_artifacts: bool
    pushed_verified: bool
    push_evidence: dict
    messages: list

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [f"PRE-REGISTRATION LOCK: {'PASS' if self.passed else 'FAIL'}",
                 f"  prereg file        : {self.prereg_path}",
                 f"  in working tree    : {self.exists_in_worktree}",
                 f"  committed          : {self.committed}"
                 + (f"  ({self.commit_sha[:12]} @ {self.commit_iso})" if self.committed else ""),
                 f"  byte-identical     : {self.identical}",
                 f"  precedes artifacts : {self.commit_precedes_artifacts}",
                 f"  pushed (public)    : {self.pushed_verified}  {self.push_evidence}"]
        for m in self.messages:
            lines.append(f"  - {m}")
        return "\n".join(lines)


def _git(*args, repo: Path | None = None):
    return subprocess.run(
        ["git", "-C", str(repo or C.REPO_ROOT), *args],
        capture_output=True, text=True, timeout=30,
    )


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _first_commit_of(path: Path, repo: Path) -> tuple[str | None, float | None, str | None]:
    """The *earliest* commit that introduced the file.

    Deliberately earliest, not latest: pre-registration is about when the
    predictions were fixed.  Using the latest commit would let an edit
    after results exist silently reset the clock -- which is the exact
    failure mode this lock exists to make impossible.
    """
    rel = path.relative_to(repo).as_posix()
    r = _git("log", "--follow", "--reverse", "--format=%H%x09%ct%x09%cI", "--", rel, repo=repo)
    if r.returncode != 0 or not r.stdout.strip():
        return None, None, None
    sha, ct, iso = r.stdout.strip().splitlines()[0].split("\t")
    return sha, float(ct), iso


def _committed_blob(path: Path, sha: str, repo: Path) -> bytes | None:
    rel = path.relative_to(repo).as_posix()
    r = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{rel}"],
                       capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


def _https_form(url: str) -> str | None:
    """``git@host:owner/repo(.git)`` / ``ssh://git@host/owner/repo`` -> https URL."""
    if not url:
        return None
    if url.startswith(("https://", "http://")):
        return url
    m = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else None


def _push_evidence(sha: str, repo: Path) -> tuple[bool, dict]:
    """Evidence that the commit reached a PUBLIC remote (§3.7).

    Two independent facts, BOTH required: the commit is an ancestor of the
    configured upstream, and that upstream answers an anonymous HTTPS
    ``ls-remote`` (credential helpers disabled for the probe, so a private
    remote cannot pass through a cached login).  A local or private commit is
    rewritable and is not a pre-registration; this used to be a warning only.
    """
    ev: dict = {}
    up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", repo=repo)
    if up.returncode != 0:
        ev["upstream"] = None
        ev["note"] = "no upstream configured; §3.7 requires a pushed public commit"
        return False, ev
    ev["upstream"] = up.stdout.strip()
    anc = _git("merge-base", "--is-ancestor", sha, ev["upstream"], repo=repo)
    ev["commit_is_ancestor_of_upstream"] = anc.returncode == 0
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo).stdout.strip()
    remote = _git("config", "--get", f"branch.{branch}.remote", repo=repo).stdout.strip()
    url = _git("remote", "get-url", remote, repo=repo).stdout.strip() if remote else ""
    ev["remote_url"] = url or None
    probe = _https_form(url)
    ev["public_probe_url"] = probe
    ev["public_anonymous_https"] = False
    if probe:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false",
                   GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
        try:
            r = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", "--exit-code",
                                probe, "HEAD"], capture_output=True, text=True,
                               timeout=60, env=env)
            ev["public_anonymous_https"] = r.returncode == 0
            if r.returncode != 0:
                ev["probe_error"] = (r.stderr or r.stdout).strip()[:160]
        except (OSError, subprocess.TimeoutExpired) as exc:  # no network, etc.
            ev["probe_error"] = f"{type(exc).__name__}: {exc}"
    else:
        ev["probe_error"] = "remote URL is not a recognised git/https form"
    return bool(anc.returncode == 0 and ev["public_anonymous_https"]), ev


def _artifact_times(artifacts) -> tuple[list, float | None]:
    rows, earliest = [], None
    for p in artifacts:
        p = Path(p)
        if not p.exists():
            continue
        t = p.stat().st_mtime
        rows.append({"path": str(p), "mtime_unix": t})
        earliest = t if earliest is None else min(earliest, t)
    return rows, earliest


def check(artifacts=(), *, repo: Path | None = None,
          prereg_path: Path | None = None) -> LockReport:
    """Evaluate the lock.  Returns a report; does not raise."""
    repo = Path(repo or C.REPO_ROOT)
    prereg = Path(prereg_path or PREREG_PATH)
    messages: list = []

    exists = prereg.exists()
    wt_sha = _sha256_bytes(prereg.read_bytes()) if exists else None
    if not exists:
        messages.append(f"{prereg} does not exist; write and commit it after Experiment D (§11)")

    sha, ct, iso = _first_commit_of(prereg, repo) if exists else (None, None, None)
    committed = sha is not None
    if exists and not committed:
        messages.append("prereg.md is untracked or uncommitted; a working-tree file is not "
                        "a pre-registration")

    blob = _committed_blob(prereg, sha, repo) if committed else None
    c_sha = _sha256_bytes(blob) if blob is not None else None
    identical = bool(c_sha and wt_sha and c_sha == wt_sha)
    if committed and not identical:
        messages.append("working-tree prereg.md differs from the committed version; the "
                        "predictions were edited after registration (§14)")

    rows, earliest = _artifact_times(artifacts)
    precedes = bool(ct is not None and (earliest is None or ct < earliest))
    if committed and not precedes and earliest is not None:
        messages.append(f"prereg commit ({iso}) is NOT earlier than the earliest Experiment E "
                        f"artifact; the predictions were registered after the results existed")

    pushed, push_ev = _push_evidence(sha, repo) if committed else (False, {})
    if committed and not pushed:
        messages.append("the prereg commit is NOT verifiably on a PUBLIC remote (§3.7): "
                        f"{push_ev}. `gh repo edit --visibility public && git push`, "
                        "then re-run; a local or private commit is rewritable and is "
                        "not a pre-registration")

    placeholders = bool(exists and "<<" in prereg.read_text(errors="replace"))
    if placeholders:
        messages.append("prereg.md still contains <<...>> template placeholders; every "
                        "field must be filled from the A/C/D/power outputs BEFORE it "
                        "is committed as the registration")

    passed = bool(exists and committed and identical and precedes and pushed
                  and not placeholders)
    return LockReport(
        passed=passed, prereg_path=str(prereg), exists_in_worktree=exists,
        committed=committed, commit_sha=sha, commit_unix=ct, commit_iso=iso,
        worktree_sha256=wt_sha, committed_sha256=c_sha, identical=identical,
        artifacts_checked=rows, earliest_artifact_unix=earliest,
        commit_precedes_artifacts=precedes, pushed_verified=pushed,
        push_evidence=push_ev, messages=messages,
    )


def require(experiment: str, artifacts=()) -> LockReport:
    """Enforce the lock for a sealed experiment.  Raises on failure.

    No override parameter exists, by design (§0.5: "do not make it skippable
    by flag").
    """
    if experiment.lower() not in LOCKED_EXPERIMENTS:
        return check(artifacts)
    rep = check(artifacts)
    if not rep.passed:
        raise PreregViolation(
            f"Experiment {experiment.upper()} analysis is sealed until the "
            f"pre-registration lock passes (§0.5).\n\n{rep.render()}"
        )
    return rep


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = check(a.artifacts)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.render())
    raise SystemExit(0 if r.passed else 1)
