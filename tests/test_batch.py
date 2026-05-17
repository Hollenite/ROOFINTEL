import unittest

from src.batch import aggregate_batch_summaries, analyze_predicted_tile
from src.data import generate_synthetic_tile
from src.estimate import SolarConfig


class BatchAnalysisTests(unittest.TestCase):
    def test_analyze_predicted_tile_returns_solar_summary_for_synthetic_mask(self):
        image, mask, transform, crs = generate_synthetic_tile(num_buildings=4, seed=7)

        summary, detail = analyze_predicted_tile(
            filename="synthetic.tif",
            image=image,
            mask=mask,
            transform=transform,
            crs=crs,
            config=SolarConfig(),
            min_area=1.0,
            simplify_tolerance=0.0,
            tariff=8.0,
            tariff_unit="INR",
        )

        self.assertEqual(summary.status, "ok")
        self.assertGreater(summary.num_roofs, 0)
        self.assertGreater(summary.total_system_kw, 0)
        self.assertGreater(summary.total_annual_kwh, 0)
        self.assertEqual(
            summary.estimated_annual_saving,
            round(summary.total_annual_kwh * 8.0, 2),
        )
        self.assertEqual(detail["aggregate"]["num_roofs"], summary.num_roofs)

    def test_aggregate_batch_summaries_counts_successful_and_failed_tiles(self):
        image, mask, transform, crs = generate_synthetic_tile(num_buildings=2, seed=3)
        ok_summary, _ = analyze_predicted_tile(
            filename="ok.tif",
            image=image,
            mask=mask,
            transform=transform,
            crs=crs,
            config=SolarConfig(),
            min_area=1.0,
            simplify_tolerance=0.0,
            tariff=10.0,
            tariff_unit="INR",
        )
        failed_summary, _ = analyze_predicted_tile(
            filename="empty.tif",
            image=image,
            mask=mask * 0,
            transform=transform,
            crs=crs,
            config=SolarConfig(),
            min_area=1.0,
            simplify_tolerance=0.0,
            tariff=10.0,
            tariff_unit="INR",
        )

        aggregate = aggregate_batch_summaries([ok_summary, failed_summary])

        self.assertEqual(aggregate["files_analyzed"], 2)
        self.assertEqual(aggregate["successful_files"], 1)
        self.assertEqual(aggregate["failed_files"], 1)
        self.assertEqual(aggregate["total_roofs"], ok_summary.num_roofs)


if __name__ == "__main__":
    unittest.main()
