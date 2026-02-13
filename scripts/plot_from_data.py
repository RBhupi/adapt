#!/usr/bin/env python3
"""External visualization script - consumes finalized pipeline data.

This script demonstrates how to use the DataAPI to:
1. Subscribe to segmentation products and poll for new data
2. Load radar grids (NetCDF) with detected cells
3. Render visualizations

**CRITICAL DESIGN PRINCIPLE:**
- Pipeline produces data -> DataStore registers -> DataAPI reads -> This script visualizes
- NO COUPLING to pipeline internals
- Can be run asynchronously, on different machines, or deleted entirely
- Pipeline doesn't know this exists

Usage
-----
Plot the latest segmentation scan::

    python scripts/plot_from_data.py --db output/adapt.db --radar KLOT --latest

Poll for new scans continuously::

    python scripts/plot_from_data.py --db output/adapt.db --radar KLOT --poll

Save figures without displaying::

    python scripts/plot_from_data.py --db output/adapt.db --radar KLOT --latest \
        --output-dir figures/ --no-show
"""

import argparse
import logging
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr

from adapt.data_access import DataAPI

logger = logging.getLogger(__name__)


def plot_segmentation(
    ds: xr.Dataset,
    output_path: Path = None,
    show: bool = True,
) -> None:
    """Render segmentation visualization.

    Parameters
    ----------
    ds : xr.Dataset
        Segmentation dataset with reflectivity and cell_labels.
    output_path : Path, optional
        If provided, save figure to this path.
    show : bool, default=True
        If True, display figure interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: Reflectivity
    ax_refl = axes[0]
    refl = ds['reflectivity'].values

    im_refl = ax_refl.imshow(
        refl,
        origin='lower',
        cmap='turbo',
        vmin=-10,
        vmax=70,
        interpolation='nearest',
    )

    ax_refl.set_title(f"Reflectivity (dBZ)\n{ds.attrs.get('scan_time', 'Unknown')}")
    ax_refl.set_xlabel("X (km)")
    ax_refl.set_ylabel("Y (km)")

    cbar_refl = plt.colorbar(im_refl, ax=ax_refl, fraction=0.046, pad=0.04)
    cbar_refl.set_label("dBZ")

    # Right panel: Segmentation + Projections
    ax_seg = axes[1]

    # Background reflectivity (faded)
    ax_seg.imshow(
        refl,
        origin='lower',
        cmap='gray',
        alpha=0.3,
        vmin=-10,
        vmax=70,
        interpolation='nearest',
    )

    # Cell labels
    if 'cell_labels' in ds:
        cell_labels = ds['cell_labels'].values
        cells_masked = np.ma.masked_where(cell_labels == 0, cell_labels)

        ax_seg.imshow(
            cells_masked,
            origin='lower',
            cmap='tab20',
            alpha=0.6,
            interpolation='nearest',
        )

        # Projection arrows (if available)
        if 'heading_x' in ds and 'heading_y' in ds:
            heading_x = ds['heading_x'].values
            heading_y = ds['heading_y'].values

            step = 20
            y_coords, x_coords = np.mgrid[
                0:heading_x.shape[0]:step, 0:heading_x.shape[1]:step
            ]

            mask = cell_labels[y_coords, x_coords] > 0

            ax_seg.quiver(
                x_coords[mask],
                y_coords[mask],
                heading_x[y_coords, x_coords][mask],
                heading_y[y_coords, x_coords][mask],
                color='red',
                alpha=0.8,
                width=0.003,
                scale=30,
            )

        n_cells = len(np.unique(cell_labels)) - 1
        ax_seg.set_title(f"Segmentation + Projections\n{n_cells} cells detected")
    else:
        ax_seg.set_title("Segmentation (no cell_labels)")

    ax_seg.set_xlabel("X (km)")
    ax_seg.set_ylabel("Y (km)")

    legend_elements = [
        mpatches.Patch(color='tab:blue', alpha=0.6, label='Detected cells'),
    ]
    ax_seg.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """External plotting entry point using DataAPI."""
    parser = argparse.ArgumentParser(
        description="Visualize pipeline data using the ADAPT DataAPI"
    )

    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='Path to the DataStore SQLite database (e.g., output/adapt.db)',
    )

    parser.add_argument(
        '--radar',
        type=str,
        required=True,
        help='Radar ID to filter by (e.g., KLOT, KDIX)',
    )

    parser.add_argument(
        '--latest',
        action='store_true',
        help='Plot the most recent segmentation file',
    )

    parser.add_argument(
        '--poll',
        action='store_true',
        help='Continuously poll for new scans and plot them',
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=10,
        help='Seconds between polls (default: 10)',
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of recent files to list/plot (default: 5)',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save rendered figures',
    )

    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display figures interactively (only save)',
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print store statistics and exit',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Initialize DataAPI
    api = DataAPI.from_path(args.db)

    # Print statistics
    if args.stats:
        stats = api.store.get_statistics(radar_id=args.radar)
        print("\n=== DataStore Statistics ===")
        print(f"Radar:          {args.radar}")
        print(f"Total products: {stats['total_products']}")
        for ptype, count in stats['products_by_type'].items():
            print(f"  {ptype}: {count}")
        sys.exit(0)

    # Figure output directory
    fig_output_dir = None
    if args.output_dir:
        fig_output_dir = Path(args.output_dir)
        fig_output_dir.mkdir(parents=True, exist_ok=True)

    # Mode: poll continuously
    if args.poll:
        sub = api.subscribe(
            product_types=["segmented_netcdf"],
            radar_id=args.radar,
        )
        logger.info(f"Polling for segmented_netcdf from {args.radar} every {args.poll_interval}s...")

        while True:
            for product in sub.poll():
                logger.info(f"New product: {product.product_id} ({product.scan_time})")
                try:
                    ds = api.read_grid(product)
                    out_path = None
                    if fig_output_dir:
                        out_path = fig_output_dir / f"{Path(product.file_path).stem}_viz.png"
                    plot_segmentation(ds, output_path=out_path, show=(not args.no_show))
                except Exception as e:
                    logger.error(f"Failed to plot {product.file_path}: {e}")
            time.sleep(args.poll_interval)

    # Mode: latest
    if args.latest:
        ds = api.get_latest_grid("segmented_netcdf", args.radar)
        if ds is None:
            logger.warning("No segmented_netcdf products found.")
            sys.exit(0)
        out_path = None
        if fig_output_dir:
            out_path = fig_output_dir / "latest_viz.png"
        plot_segmentation(ds, output_path=out_path, show=(not args.no_show))
        sys.exit(0)

    # Mode: list recent
    products = api.list_grids(
        product_type="segmented_netcdf",
        radar_id=args.radar,
        limit=args.limit,
    )

    if not products:
        logger.warning("No segmentation products found. Has the pipeline been run?")
        sys.exit(0)

    logger.info(f"Found {len(products)} segmentation product(s)")

    for i, product in enumerate(products):
        logger.info(f"Loading {product.file_path}...")
        try:
            ds = api.read_grid(product)
            out_path = None
            if fig_output_dir:
                out_path = fig_output_dir / f"{Path(product.file_path).stem}_viz.png"
            plot_segmentation(ds, output_path=out_path, show=(not args.no_show))
            logger.info(f"Processed ({i + 1}/{len(products)})")
        except Exception as e:
            logger.error(f"Failed to plot {product.file_path}: {e}")
            continue

    logger.info("Visualization complete.")


if __name__ == "__main__":
    main()
