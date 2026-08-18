"""Loader and validator for the reproduced MultiWorld §B.2 protocol (§0.6).

The point of this module is that no faithfulness-critical hyperparameter can
reach a training run without a recorded provenance.  ``load()`` raises unless
every field declares a status, and every field that is not straight from a
paper carries a justification and a decider.  ``summary_table()`` is printed at
the top of every Architecture A run and copied into ``metadata.json``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

import config as C

PROTOCOL_PATH = C.REPO_ROOT / "protocol" / "multiworld_b2.yaml"

VALID_STATUS = ("PAPER", "VPT", "ADAPTED", "DECIDED")
#: Statuses that require an explicit human-recorded rationale.
NEEDS_JUSTIFICATION = ("ADAPTED", "DECIDED")


class ProtocolError(RuntimeError):
    """Raised when the protocol record is incomplete -- a STOP condition (§0.6)."""


@dataclass(frozen=True)
class Protocol:
    fields: dict          # name -> {"value": ..., "status": ..., ...}
    source: dict
    sha256: str
    path: str

    def __getitem__(self, key: str):
        try:
            return self.fields[key]["value"]
        except KeyError as exc:
            raise ProtocolError(
                f"protocol field {key!r} is not recorded in {self.path}. "
                f"Add it with a provenance status rather than hardcoding it."
            ) from exc

    def get(self, key: str, default=None):
        return self.fields[key]["value"] if key in self.fields else default

    def status(self, key: str) -> str:
        return self.fields[key]["status"]

    def unreviewed(self) -> list:
        """Fields decided by the implementing agent and not yet human-reviewed."""
        return sorted(
            k for k, v in self.fields.items()
            if v.get("decided_by") == "implementing-agent"
        )

    def summary_table(self) -> str:
        w = max(len(k) for k in self.fields) + 2
        lines = [
            f"MultiWorld §B.2 protocol record  ({self.path}, sha256={self.sha256[:12]})",
            f"  source: {self.source.get('paper')} arXiv:{self.source.get('arxiv')} "
            f"{self.source.get('section')}",
            f"  {'field'.ljust(w)}{'status'.ljust(9)}value",
        ]
        for k in sorted(self.fields):
            v = self.fields[k]
            lines.append(f"  {k.ljust(w)}{str(v['status']).ljust(9)}{v['value']!r}")
        un = self.unreviewed()
        if un:
            lines.append(
                f"  !! {len(un)} field(s) decided by the implementing agent and NOT "
                f"human-reviewed: {', '.join(un)}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "source": self.source,
            "fields": self.fields,
            "unreviewed_fields": self.unreviewed(),
        }


def load(path: Path | None = None, *, require_reviewed: bool = False) -> Protocol:
    """Load and validate the protocol record.

    ``require_reviewed=True`` additionally refuses any field still marked
    ``decided_by: implementing-agent``.  Not the default, because that would
    block the whole pipeline before a human has ever seen it; the fields are
    instead printed loudly at every run.
    """
    p = Path(path or PROTOCOL_PATH)
    if not p.exists():
        raise ProtocolError(
            f"{p} not found.  §0.6: the standard action-following protocol must be "
            f"reproduced from MultiWorld §B.2, not guessed.  STOP and supply the record."
        )
    raw_bytes = p.read_bytes()
    doc = yaml.safe_load(raw_bytes)
    fields = doc.get("fields") or {}
    if not fields:
        raise ProtocolError(f"{p} declares no fields")

    problems = []
    for name, spec in fields.items():
        if not isinstance(spec, dict) or "value" not in spec:
            problems.append(f"{name}: missing `value`")
            continue
        status = spec.get("status")
        if status not in VALID_STATUS:
            problems.append(f"{name}: status {status!r} not in {VALID_STATUS}")
            continue
        if status in NEEDS_JUSTIFICATION:
            if not str(spec.get("justification", "")).strip():
                problems.append(f"{name}: status {status} requires a `justification`")
            if not str(spec.get("decided_by", "")).strip():
                problems.append(f"{name}: status {status} requires a `decided_by`")
        if status == "PAPER" and not str(spec.get("quote", "")).strip():
            problems.append(f"{name}: status PAPER requires the supporting `quote`")
    if problems:
        raise ProtocolError(
            "protocol record is incomplete (§0.6 forbids silent substitution):\n  - "
            + "\n  - ".join(problems)
        )

    proto = Protocol(
        fields=fields,
        source=doc.get("source", {}),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=str(p),
    )

    if require_reviewed and proto.unreviewed():
        raise ProtocolError(
            "these protocol fields are still the implementing agent's decision and "
            "have not been human-reviewed: " + ", ".join(proto.unreviewed())
        )

    # Cross-checks against config, so the two cannot silently drift apart.
    if int(proto["clip_length"]) != int(C.CLIP_LENGTH):
        raise ProtocolError(
            f"protocol clip_length={proto['clip_length']} but config.CLIP_LENGTH="
            f"{C.CLIP_LENGTH}"
        )
    if int(proto["image_resolution"]) != int(C.RENDER_SIZE):
        raise ProtocolError(
            f"protocol image_resolution={proto['image_resolution']} but "
            f"config.RENDER_SIZE={C.RENDER_SIZE}"
        )
    return proto


if __name__ == "__main__":
    print(load().summary_table())
