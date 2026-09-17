"""Checkpointed batch ingestion pipeline for field-level geospatial data.

Adapted from the geoai ``GlobStep`` + ``Pipeline`` framework: a small
step-based harness that expands per-field-year directories into individual
work items, runs registration / alignment steps (field boundary alignment,
yield-monitor raster registration), and checkpoints progress so a large
multi-field dataset can be processed incrementally and resumed after
failures. This is the batch-ingestion half of the field-level geospatial
data pipeline (gap g6): the per-field work items produced here feed the
field-level loaders (``field_roi``, ``nearest_reproject``,
``field_patch_dataset``) that register each layer to the field geometry.

The ``on_error`` policy (``"skip"`` vs ``"fail"``) and the
``CheckpointManager`` resume behaviour support the missing-modality
robustness goal: a field whose boundary or yield raster is absent can be
skipped without aborting the whole multi-field run, and a partially
processed dataset can be resumed from the last completed item.
"""

import glob as _glob
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from dataset.spectral_indices import compute_evi, compute_ndvi

#: Default raster extensions expanded by :class:`FieldGlobStep`.
DEFAULT_RASTER_EXTENSIONS: List[str] = [".tif", ".tiff", ".jp2", ".img"]


class PipelineStep:
    """Base class for a single stage of a field-level ingestion pipeline.

    A step receives one work item (a ``dict``) and returns a (possibly
    modified) item. Steps that fan one item out into many (e.g. globbing a
    directory into per-file items) override :meth:`expand` instead.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single work item in place."""
        return item

    def expand(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optionally fan one item out into several work items."""
        return [item]


class FieldGlobStep(PipelineStep):
    """Expand a field-year directory or glob pattern into individual work items.

    Typically the first step in a pipeline. Takes a single item with an
    ``input_dir`` or ``input_pattern`` key and yields multiple items, one per
    matched raster file. Each emitted item carries the matched file under
    ``input_path`` so downstream registration steps know which layer to align
    to the field geometry.

    Parameters
    ----------
    name : str
        Step name (default ``"glob"``).
    extensions : sequence of str, optional
        Raster extensions to match when ``input_dir`` is given. Defaults to
        :data:`DEFAULT_RASTER_EXTENSIONS`.
    """

    def __init__(
        self,
        name: str = "glob",
        extensions: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name)
        self.extensions = list(extensions) if extensions else DEFAULT_RASTER_EXTENSIONS

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return item

    def expand(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_dir = item.get("input_dir")
        input_pattern = item.get("input_pattern")
        if input_pattern:
            files = sorted(_glob.glob(input_pattern))
        elif input_dir:
            files = []
            for ext in self.extensions:
                files.extend(_glob.glob(os.path.join(input_dir, f"*{ext}")))
            files = sorted(set(files))
        else:
            raise ValueError("Item must have 'input_dir' or 'input_pattern' key")

        items = []
        for f in files:
            new_item = dict(item)
            new_item["input_path"] = f
            new_item.pop("input_dir", None)
            new_item.pop("input_pattern", None)
            items.append(new_item)
        return items


class FieldFunctionStep(PipelineStep):
    """Apply an arbitrary callable to each work item.

    Used for registration / alignment steps such as field boundary alignment
    or yield-monitor raster registration: the callable receives the item dict
    and returns the (possibly enriched) item dict.

    Parameters
    ----------
    func : callable
        ``func(item: dict) -> dict`` applied to each work item.
    name : str
        Step name (default ``"function"``).
    """

    def __init__(self, func: Callable[[Dict[str, Any]], Dict[str, Any]], name: str = "function") -> None:
        super().__init__(name)
        self.func = func

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.func(item)


class FieldSpectralIndexStep(PipelineStep):
    """Derive vegetation indices (NDVI / EVI) from an optical reflectance raster.

    This is the optical-imagery preprocessing stage of the field-level
    geospatial pipeline (gap g6): it converts raw NIR/Red/Blue surface
    reflectance bands into vegetation-index rasters that feed the optical
    vision encoder (gap g2). The step reads the raster at ``item["input_path"]``
    (a ``(C, H, W)`` channel-first stack), computes NDVI and EVI from the
    configured band indices, and writes the derived index rasters next to the
    source file as ``<stem>_NDVI.tif`` / ``<stem>_EVI.tif``. The item is
    enriched with the output paths under ``item["spectral_indices"]``.

    Parameters
    ----------
    name : str
        Step name (default ``"spectral_indices"``).
    nir_idx, red_idx, blue_idx : int
        Channel indices of the NIR / Red / Blue bands in the source raster.
    """

    def __init__(self, name: str = "spectral_indices",
                 nir_idx: int = 0, red_idx: int = 1, blue_idx: int = 2) -> None:
        super().__init__(name)
        self.nir_idx = nir_idx
        self.red_idx = red_idx
        self.blue_idx = blue_idx

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        import rasterio
        from rasterio.transform import from_origin

        src_path = item["input_path"]
        with rasterio.open(src_path) as src:
            stack = src.read().astype(np.float32)
            profile = src.profile
            transform = src.transform

        ndvi = compute_ndvi(None, None, stack=stack,
                            nir_idx=self.nir_idx, red_idx=self.red_idx)
        evi = compute_evi(None, None, None, stack=stack,
                          nir_idx=self.nir_idx, red_idx=self.red_idx,
                          blue_idx=self.blue_idx)

        stem = os.path.splitext(src_path)[0]
        out_paths = {}
        for name, arr in (("NDVI", ndvi), ("EVI", evi)):
            out_path = f"{stem}_{name}.tif"
            with rasterio.open(
                out_path, "w", driver="GTiff", height=arr.shape[0],
                width=arr.shape[1], count=1, dtype="float32",
                crs=profile.get("crs"), transform=transform,
            ) as dst:
                dst.write(arr, 1)
            out_paths[name] = out_path

        item["spectral_indices"] = out_paths
        return item


class CheckpointManager:
    """Persist completed work-item keys so a run can resume after a failure.

    Completed item keys are appended to a JSON lines file. On resume, items
    whose key is already recorded are skipped, so a large multi-field dataset
    can be processed incrementally without redoing finished work.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint file (JSON lines).
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._done: set = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._done.add(json.loads(line))

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        self._done.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(key) + "\n")


class FieldPipeline:
    """Checkpointed, step-based batch ingestion pipeline for field-level data.

    Runs a sequence of :class:`PipelineStep` objects over a list of input
    items. Steps that override :meth:`PipelineStep.expand` fan items out (e.g.
    :class:`FieldGlobStep`); other steps transform items in place. Progress is
    checkpointed via :class:`CheckpointManager` so the pipeline can be resumed
    after failures, and the ``on_error`` policy controls whether a failing
    item is skipped or aborts the run.

    Parameters
    ----------
    steps : sequence of PipelineStep
        Ordered steps to apply to each work item.
    checkpoint_path : str, optional
        Path to the resume checkpoint file. If ``None``, no checkpointing is
        performed.
    on_error : str
        ``"skip"`` to log and continue past a failing item, or ``"fail"`` to
        re-raise the first error (default ``"skip"``).
    key_fn : callable, optional
        ``key_fn(item: dict) -> str`` producing a stable per-item key used for
        checkpointing. Defaults to ``item["input_path"]``.
    """

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        checkpoint_path: Optional[str] = None,
        on_error: str = "skip",
        key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self.steps = list(steps)
        self.on_error = on_error
        self.key_fn = key_fn or (lambda item: str(item.get("input_path", "")))
        self.checkpoint = CheckpointManager(checkpoint_path) if checkpoint_path else None

    def run(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute the pipeline over ``items`` and return the processed items.

        Each input item is expanded through the step chain. Items whose
        checkpoint key is already recorded are skipped on resume. Depending on
        ``on_error``, a failing item is either skipped (logged) or raises.
        """
        results: List[Dict[str, Any]] = []
        for item in tqdm(items, desc="field-pipeline"):
            key = self.key_fn(item)
            if self.checkpoint is not None and self.checkpoint.is_done(key):
                continue
            try:
                current = [item]
                for step in self.steps:
                    expanded = []
                    for it in current:
                        expanded.extend(step.expand(it))
                    current = [step.process(it) for it in expanded]
                results.extend(current)
                if self.checkpoint is not None:
                    self.checkpoint.mark_done(key)
            except Exception as exc:  # noqa: BLE001 - policy-controlled
                if self.on_error == "fail":
                    raise
                print(f"FieldPipeline skipping item {key}: {exc}")
        return results


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for field in ("field_2021", "field_2022"):
            (tmp / field).mkdir()
            for layer in ("DEM", "S2", "yield"):
                (tmp / field / f"{layer}.tif").touch()

        def register(item: Dict[str, Any]) -> Dict[str, Any]:
            item["registered"] = True
            return item

        pipeline = FieldPipeline(
            steps=[
                FieldGlobStep(extensions=[".tif"]),
                FieldFunctionStep(register, name="register"),
            ],
            checkpoint_path=str(tmp / "checkpoint.jsonl"),
            on_error="skip",
        )
        out = pipeline.run([{"input_dir": str(tmp / "field_2021")}, {"input_dir": str(tmp / "field_2022")}])
        print("processed items:", len(out))
        print("all registered:", all(it["registered"] for it in out))
        print("sample:", out[0]["input_path"])
