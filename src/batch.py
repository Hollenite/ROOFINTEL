"""Batch analysis helpers for B2B GeoTIFF portfolio screening."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.confidence import compute_tile_confidence, confidence_to_dict
from src.estimate import SolarConfig, estimate_all_roofs
from src.feasibility import enrich_polygons_with_feasibility
from src.utils import alignment_debug_info, validate_polygon_raster_alignment
from src.vectorize import mask_to_polygons


@dataclass
class BatchTileSummary:
    """Compact, CSV-friendly result for one uploaded imagery tile."""

    filename: str
    status: str
    message: str
    num_roofs: int = 0
    total_roof_area: float = 0.0
    total_roof_area_unit: str = ""
    total_system_kw: float = 0.0
    total_annual_kwh: float = 0.0
    estimated_annual_saving: float = 0.0
    tariff: float = 0.0
    tariff_unit: str = ""
    confidence_score: float = 0.0
    warnings_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def analyze_predicted_tile(
    *,
    filename: str,
    image: np.ndarray,
    mask: np.ndarray,
    transform,
    crs,
    config: SolarConfig,
    min_area: float,
    simplify_tolerance: float,
    tariff: float,
    tariff_unit: str,
) -> Tuple[BatchTileSummary, Dict]:
    """Analyze one image and predicted mask, returning summary + detail."""
    warnings_list: List[str] = []
    use_pixel_coords = crs is None
    if crs is None:
        warnings_list.append("No CRS detected; area uses pixel-coordinate fallback.")

    polygons = mask_to_polygons(
        mask,
        transform=transform,
        crs=crs,
        min_area=min_area,
        simplify_tolerance=simplify_tolerance,
        use_pixel_coords=use_pixel_coords,
    )
    if not polygons:
        summary = BatchTileSummary(
            filename=filename,
            status="no_roofs",
            message="No roof polygons found after vectorization.",
            tariff=tariff,
            tariff_unit=tariff_unit,
            warnings_count=len(warnings_list),
        )
        return summary, {"warnings": warnings_list, "polygons": [], "per_roof": []}

    if crs is not None and transform is not None:
        raster_bounds = (
            transform.c,
            transform.f + transform.e * mask.shape[0],
            transform.c + transform.a * mask.shape[1],
            transform.f,
        )
        _, align_warnings = validate_polygon_raster_alignment(
            raster_bounds,
            [p["geometry"] for p in polygons],
        )
        warnings_list.extend(align_warnings)

    align_debug = alignment_debug_info(transform, mask.shape, polygons, crs)
    polygons = enrich_polygons_with_feasibility(polygons, image=image, mask=mask)
    per_roof, aggregate = estimate_all_roofs(polygons, config)

    tile_conf = compute_tile_confidence(
        imagery_date=None,
        num_polygons=len(polygons),
        min_area_used=min_area,
        overlap_ratio=align_debug.get("overlap_ratio") if align_debug else None,
        alignment_warnings=[w for w in warnings_list if "overlap" in w.lower()],
        has_crs=(crs is not None),
    )
    estimated_annual_saving = round(aggregate["total_annual_kwh"] * tariff, 2)

    summary = BatchTileSummary(
        filename=filename,
        status="ok",
        message="Analyzed successfully.",
        num_roofs=aggregate["num_roofs"],
        total_roof_area=aggregate["total_roof_area"],
        total_roof_area_unit=aggregate["total_roof_area_unit"],
        total_system_kw=aggregate["total_system_kw"],
        total_annual_kwh=aggregate["total_annual_kwh"],
        estimated_annual_saving=estimated_annual_saving,
        tariff=tariff,
        tariff_unit=tariff_unit,
        confidence_score=tile_conf.overall_confidence_score or 0.0,
        warnings_count=len(warnings_list),
    )
    detail = {
        "warnings": warnings_list,
        "polygons": polygons,
        "per_roof": per_roof,
        "aggregate": aggregate,
        "confidence": confidence_to_dict(tile_conf),
    }
    return summary, detail


def aggregate_batch_summaries(summaries: List[BatchTileSummary]) -> Dict:
    """Aggregate successful tile summaries into portfolio-level totals."""
    successful = [s for s in summaries if s.status == "ok"]
    return {
        "files_analyzed": len(summaries),
        "successful_files": len(successful),
        "failed_files": len(summaries) - len(successful),
        "total_roofs": sum(s.num_roofs for s in successful),
        "total_system_kw": round(sum(s.total_system_kw for s in successful), 2),
        "total_annual_kwh": round(sum(s.total_annual_kwh for s in successful), 1),
        "estimated_annual_saving": round(
            sum(s.estimated_annual_saving for s in successful),
            2,
        ),
    }
