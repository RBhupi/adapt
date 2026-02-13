#!/usr/bin/env python3
"""``Adapt`` NEXRAD Radar Processing Pipeline Runner.

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar-id KHTX
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --mode historical

Note: User config in scripts/user_config.py, do not change config in src/param_config.py

Architecture:
    Thread 1 (Orchestration): Downloader -> Processor -> Repository writes
    Thread 2 (PlotConsumer): Polls Repository -> Loads artifacts -> Plots -> Saves

The PlotConsumer is completely decoupled from processing. It polls the
DataRepository for new analysis artifacts and generates visualizations
independently. Repository is the only synchronization boundary.

Author: Bhupendra Raut
"""

import sys
import argparse
import importlib.util
import threading
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from adapt.schemas import init_runtime_config
from adapt.pipeline.orchestrator import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run the ADAPT NEXRAD processing pipeline")
    parser.add_argument("config", help="Path to user config file")
    parser.add_argument("--radar-id", help="Override radar ID")
    parser.add_argument("--mode", choices=["realtime", "historical"], help="Override mode")
    parser.add_argument("--start-time", help="Start time (ISO format)")
    parser.add_argument("--end-time", help="End time (ISO format)")
    parser.add_argument("--base-dir", help="Output directory")
    parser.add_argument("--max-runtime", type=int, help="Max runtime in minutes (realtime)")
    parser.add_argument("--rerun", action="store_true", help="Delete output directories before running")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot consumer thread")
    parser.add_argument("--plot-interval", type=float, default=2.0, help="Plot polling interval in seconds")
    parser.add_argument("--show-plots", action="store_true", help="Display plots in live window")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    
    # Complete runtime initialization - does EVERYTHING
    config = init_runtime_config(args)
    
    # Create orchestrator with ready config - pure execution engine
    orchestrator = PipelineOrchestrator(config)
    
    # Print summary using args directly where possible
    print(f"\n{'='*60}")
    print("ADAPT Radar Processing Pipeline")
    print('='*60)
    print(f"Config: {args.config}")
    print(f"Radar:  {args.radar_id or config.downloader.radar_id}")  # Use args first
    print(f"Mode:   {args.mode or config.mode}")  # Use args first  
    print(f"Output: {args.base_dir or config.base_dir}")  # Use args first
    print(f"Run ID: {config.run_id}")
    print('='*60)

    if args.verbose:
        import json
        print("\nFull Runtime Configuration:")
        print(json.dumps(config.model_dump(), indent=2, default=str))
        print('='*60)

    # Plot consumer thread (repository-driven, decoupled from processing)
    plot_consumer = None
    stop_event = threading.Event()

    if not args.no_plot:
        # Import here to avoid circular imports and keep processing decoupled
        from adapt.visualization.plotter import PlotConsumer

        # We need to start orchestrator first to initialize repository
        # Then start plot consumer with the same repository reference
        print("Starting pipeline with plot consumer...")
    else:
        print("Starting pipeline (plotting disabled)...")

    # Start orchestrator in a separate thread so we can manage plot consumer
    orchestrator_thread = threading.Thread(
        target=_run_orchestrator,
        args=(orchestrator, args.max_runtime, stop_event),
        name="OrchestratorRunner",
        daemon=False
    )
    orchestrator_thread.start()

    # Wait briefly for repository to be initialized
    time.sleep(0.5)

    # Start plot consumer if enabled
    if not args.no_plot and orchestrator.repository is not None:
        from adapt.visualization.plotter import PlotConsumer

        radar_id = args.radar_id or config.downloader.radar_id  # Use args first
        plot_output_dir = Path(config.output_dirs["base"]) / radar_id / "plots"

        plot_consumer = PlotConsumer(
            repository=orchestrator.repository,
            stop_event=stop_event,
            output_dir=plot_output_dir,
            config=config,
            poll_interval=args.plot_interval,
            show_live=args.show_plots,
            name="PlotConsumer"
        )
        plot_consumer.start()
        print(f"Plot consumer started (polling every {args.plot_interval}s)")

    try:
        # Wait for orchestrator to complete
        orchestrator_thread.join()
    except KeyboardInterrupt:
        print("\nShutdown signal received...")
    finally:
        # Signal plot consumer to stop
        stop_event.set()

        # Wait for plot consumer to finish current work
        if plot_consumer is not None and plot_consumer.is_alive():
            print("Waiting for plot consumer to finish...")
            plot_consumer.join(timeout=10)
            if plot_consumer.is_alive():
                print("Warning: Plot consumer did not stop cleanly")

        print("Pipeline shutdown complete.")


def _run_orchestrator(orchestrator: PipelineOrchestrator, max_runtime: int, stop_event: threading.Event):
    """Run orchestrator and signal stop event when done."""
    try:
        orchestrator.start(max_runtime=max_runtime)
    finally:
        # Signal plot consumer to stop when orchestrator is done
        stop_event.set()


if __name__ == "__main__":
    main()
