# SPDX-License-Identifier: AGPL-3.0-only
"""Banc-realite paths and runtime configuration."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from .dossiers import BANC_VAR, dossier_courant

BANC_ROOT = BANC_VAR
WORKTREE = Path(__file__).resolve().parents[2]
_GREENLOOP = dossier_courant("greenloop")
_MOREAU = dossier_courant("moreau")
AREA = _GREENLOOP.area_root
GOLD = _GREENLOOP.gold_dir
CORPUS = _GREENLOOP.corpus_dir
MOREAU_INGESTION_SORTIES = BANC_ROOT / "dossiers" / "moreau" / "ingestion"
MOREAU_INGESTION_MAX_CALLS = 80
MOREAU_INGESTION_MAX_EMBEDDINGS = 3000
SORTIES = BANC_ROOT / "sorties"
INGESTION2_SORTIES = SORTIES / "ingestion-2"
INGESTION2_MAX_CALLS = 130
INGESTION3_SORTIES = SORTIES / "ingestion-3"
INGESTION3_REPRISE_LOG = INGESTION3_SORTIES / "ingestion-reprise.jsonl"
INGESTION3_MAX_CALLS = 130
INGESTION3_MAX_EMBEDDINGS = 12000
SCHEMA_PRODUCTION = BANC_ROOT / "entrees" / "schema-aire-production.sql"
INSTANCES_PATH = BANC_ROOT / "instances.yaml"
SUPABASE_CREDENTIALS_PATH = BANC_ROOT / "supabase-credentials.yaml"
RELECTURE_VERDICTS = SORTIES / "relecture-verdicts-claude.json"
ARRET = BANC_ROOT / "_etage2-arret.json"
RESULT = BANC_ROOT / "_etage2-result.json"
REPORT = BANC_ROOT / "RAPPORT-ETAGE2-BRAS-C.md"
VAULT_REF = _GREENLOOP.vault_ref
INSTANCE = _GREENLOOP.instance
TABULAR_SKILL = "tabulaire-deterministe"
BANC_ACTOR_ID = uuid.UUID("00000000-0000-4000-8000-b00000000001")
SMOKE_QUESTION_ID = _GREENLOOP.smoke_question_id


def export_runtime_env() -> None:
    """Point Brain tooling at the private banc cluster (never production)."""
    os.environ.setdefault("BRAIN_CLAIMING_TABULAR", "1")
    os.environ["BRAIN_INSTANCES_PATH"] = str(INSTANCES_PATH)
    os.environ["BRAIN_SUPABASE_CREDENTIALS_PATH"] = str(SUPABASE_CREDENTIALS_PATH)
    secrets = BANC_ROOT / "secrets.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if "=" in line and not os.environ.get(line.split("=", 1)[0]):
                key, value = line.split("=", 1)
                os.environ[key] = value
    try:
        from brain.instances import reload_instance_registry
    except ImportError:
        reload_instance_registry = None  # type: ignore[misc,assignment]

    reload_instance_registry(INSTANCES_PATH)


def _instance_area_roots() -> dict[str, Path]:
    """Area roots par instance — GreenLoop inchangé, MOREAU dédié."""
    return {
        dossier.instance: dossier.area_root
        for dossier in (dossier_courant("greenloop"), dossier_courant("moreau"))
    }


def write_banc_config_files(*, preserve_existing_areas: bool = True) -> None:
    """Create banc-local instances + credentials YAML (not production paths)."""
    secrets = BANC_ROOT / "secrets.env"
    service_key = ""
    url = "http://127.0.0.1:56100"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            if line.startswith("BANC_BRAIN_SERVICE_KEY="):
                service_key = line.split("=", 1)[1].strip()
            elif line.startswith("BRAIN_POSTGREST_URL="):
                url = line.split("=", 1)[1].strip()
    area_roots = _instance_area_roots()
    if preserve_existing_areas:
        for area_root in area_roots.values():
            area_root.mkdir(parents=True, exist_ok=True)
    lines = ["include_defaults: false", "instances:"]
    for instance, area_root in sorted(area_roots.items()):
        lines.append(f"  {instance}:")
        lines.append(f"    area_root: {area_root}")
    INSTANCES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cred_lines = ["instances:"]
    for instance in sorted(area_roots):
        cred_lines.append(f"  {instance}:")
        cred_lines.append(f"    url: {url}")
        cred_lines.append(f"    service_role_key: {service_key}")
    SUPABASE_CREDENTIALS_PATH.write_text("\n".join(cred_lines) + "\n", encoding="utf-8")
    os.chmod(SUPABASE_CREDENTIALS_PATH, 0o600)
