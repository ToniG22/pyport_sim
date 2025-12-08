"""Run all test configurations."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_boat_charger import run_simulation as run_boat_charger
from tests.test_boat_charger_pv import run_simulation as run_boat_charger_pv
from tests.test_boat_charger_bess import run_simulation as run_boat_charger_bess
from tests.test_boat_charger_pv_bess import run_simulation as run_boat_charger_pv_bess


def run_all_tests(use_optimizer: bool = None):
    """
    Run all test configurations.
    
    Args:
        use_optimizer: If None, run both with and without optimizer.
                      If True/False, run only that configuration.
    """
    test_configs = [
        ("1 Boat, 1 Charger", run_boat_charger),
        ("1 Boat, 1 Charger, 1 PV", run_boat_charger_pv),
        ("1 Boat, 1 Charger, 1 BESS", run_boat_charger_bess),
        ("1 Boat, 1 Charger, 1 PV, 1 BESS", run_boat_charger_pv_bess),
    ]
    
    optimizer_modes = []
    if use_optimizer is None:
        optimizer_modes = [False, True]
    else:
        optimizer_modes = [use_optimizer]
    
    total_tests = len(test_configs) * len(optimizer_modes)
    current_test = 0
    
    print("\n" + "=" * 70)
    print("RUNNING ALL TEST CONFIGURATIONS")
    print(f"Total tests: {total_tests}")
    print("=" * 70 + "\n")
    
    for opt_mode in optimizer_modes:
        opt_str = "WITH OPTIMIZER" if opt_mode else "WITHOUT OPTIMIZER"
        print(f"\n{'#' * 70}")
        print(f"# {opt_str}")
        print(f"{'#' * 70}\n")
        
        for name, run_func in test_configs:
            current_test += 1
            print(f"\n[Test {current_test}/{total_tests}] {name}")
            print("-" * 50)
            try:
                run_func(use_optimizer=opt_mode)
                print(f"✓ PASSED: {name} ({opt_str})")
            except Exception as e:
                print(f"✗ FAILED: {name} ({opt_str})")
                print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all port simulation tests")
    parser.add_argument(
        "--optimizer",
        action="store_true",
        help="Run only with optimizer enabled",
    )
    parser.add_argument(
        "--no-optimizer",
        action="store_true",
        help="Run only without optimizer",
    )
    
    args = parser.parse_args()
    
    if args.optimizer and args.no_optimizer:
        print("Error: Cannot specify both --optimizer and --no-optimizer")
        sys.exit(1)
    elif args.optimizer:
        run_all_tests(use_optimizer=True)
    elif args.no_optimizer:
        run_all_tests(use_optimizer=False)
    else:
        # Run both modes
        run_all_tests(use_optimizer=None)
