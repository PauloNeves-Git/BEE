"""Utilities for detecting and applying deprotections on product SMILES."""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm


def _load_protection_groups(json_source: str | Path | Iterable[dict]) -> List[dict]:
    """Load protection group definitions from a path or iterable of dicts."""
    if isinstance(json_source, (str, Path)):
        with open(json_source, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return list(json_source)


def _compile_protection_groups(base_groups: List[dict]) -> List[dict]:
    """Compile SMARTS/reaction patterns from a list of group dicts."""

    compiled = []
    for group in base_groups:
        smarts = group.get("smarts")
        deprotecting_smirks = group.get("deprotecting_smirks")
        if not smarts or not deprotecting_smirks:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        reaction = AllChem.ReactionFromSmarts(deprotecting_smirks)
        if pattern is None or reaction is None:
            continue
        compiled.append({"id": group.get("id"), "pattern": pattern, "reaction": reaction})
    return compiled


def _process_smiles_factory(protection_groups: List[dict]) -> Callable[[str], Tuple[str, str]]:
    """Create a SMILES processor that applies deprotections."""

    def _process_smiles(smiles: str) -> Tuple[str, str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, "no deprotection"

        for group in protection_groups:
            if mol.HasSubstructMatch(group["pattern"]):
                outcomes = group["reaction"].RunReactants((mol,))
                if outcomes:
                    first_product = outcomes[0][0]
                    return Chem.MolToSmiles(first_product, isomericSmiles=True), str(group["id"])
        return smiles, "no deprotection"

    return _process_smiles


def _process_slice_in_process(
    offset: int,
    smiles_slice: List[str],
    base_groups: List[dict],
    progress_batch_size: int,
) -> Tuple[List[Tuple[int, Tuple[str, str]]], List[int]]:
    """Process a slice inside a separate process, capturing progress locally."""

    compiled_groups = _compile_protection_groups(list(base_groups))
    process_smiles = _process_smiles_factory(compiled_groups)

    results: List[Tuple[int, Tuple[str, str]]] = []
    progress_log: List[int] = []
    processed_since_update = 0

    for local_idx, smiles in enumerate(smiles_slice):
        results.append((offset + local_idx, process_smiles(smiles)))
        processed_since_update += 1
        if processed_since_update >= progress_batch_size:
            progress_log.append(processed_since_update)
            processed_since_update = 0

    if processed_since_update:
        progress_log.append(processed_since_update)

    return results, progress_log


def apply_deprotections(
    df: pd.DataFrame,
    protections_json: str | Path | Iterable[dict],
    stage: str,
    n_jobs: int = 1,
    show_progress: bool = True,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Detect protection groups and apply deprotection reactions to products.

    Args:
        df: Input DataFrame containing a ``product_smiles`` column.
        protections_json: Path to a JSON file or an iterable of protection group
            dictionaries with ``smarts``, ``deprotecting_smirks``, and ``id``.
        stage: Stage label used to name the resulting columns.
        n_jobs: Number of worker processes to use for processing. Values
            greater than 1 enable multiprocessing to speed up large datasets.
        show_progress: Whether to display a tqdm progress bar.
        chunksize: Batch size used for emitting progress updates from workers.
            Defaults to 1,000 if not provided.

    Returns:
        DataFrame copy with two new columns:
        ``f"{stage}_deprotected_smiles"`` and ``f"{stage}_deprotection_id"``.
    """

    base_groups = _load_protection_groups(protections_json)

    smiles_list = df["product_smiles"].tolist()
    total = len(smiles_list)
    if total == 0:
        updated_df = df.copy()
        updated_df[f"{stage}_deprotected_smiles"] = []
        updated_df[f"{stage}_deprotection_id"] = []
        return updated_df

    deprotected: List[str | None] = [None] * total
    deprotection_ids: List[str | None] = [None] * total

    if n_jobs and n_jobs > 1:
        workers = max(1, n_jobs)
        per_worker = math.ceil(total / workers)
        progress_batch_size = max(1, chunksize or 1000)

        with tqdm(
            total=total,
            disable=not show_progress,
            desc="Applying deprotections",
        ) as pbar:
            progress_callback = pbar.update if show_progress else None

            task_inputs: List[Tuple[int, List[str], List[dict], int]] = []
            for worker_idx in range(workers):
                start = worker_idx * per_worker
                end = min(total, start + per_worker)
                slice_smiles = smiles_list[start:end]
                if not slice_smiles:
                    continue
                task_inputs.append((start, slice_smiles, list(base_groups), progress_batch_size))

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process_slice_in_process, *args) for args in task_inputs]
                for future in as_completed(futures):
                    results, progress_log = future.result()
                    if progress_callback:
                        for processed in progress_log:
                            progress_callback(processed)
                    for idx, (deprot_smiles, deprot_id) in results:
                        deprotected[idx] = deprot_smiles
                        deprotection_ids[idx] = deprot_id
    else:
        compiled_groups = _compile_protection_groups(base_groups)
        _process_smiles = _process_smiles_factory(compiled_groups)
        for idx, smiles in enumerate(
            tqdm(
                smiles_list,
                total=len(smiles_list),
                disable=not show_progress,
                desc="Applying deprotections",
            )
        ):
            deprot_smiles, deprot_id = _process_smiles(smiles)
            deprotected[idx] = deprot_smiles
            deprotection_ids[idx] = deprot_id

    updated_df = df.copy()
    updated_df[f"{stage}_deprotected_smiles"] = deprotected
    updated_df[f"{stage}_deprotection_id"] = deprotection_ids
    return updated_df
