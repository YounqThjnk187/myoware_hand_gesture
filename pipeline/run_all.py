"""Run the full pipeline (Steps 1-9) in order. Step 5 is fitted inside Step 6."""

import argparse

from pipeline.step1_data_inspection import inspect_data
from pipeline.step2_filtering import run_filtering
from pipeline.step3_windowing import run_windowing
from pipeline.step4_features import run_features
from pipeline.step6_splitting import split_and_normalize
from pipeline.step7_training import train
from pipeline.step8_evaluation import evaluate
from pipeline.step9_deployment import export_c

STEPS = (
    (1, inspect_data.main),
    (2, run_filtering.main),
    (3, run_windowing.main),
    (4, run_features.main),
    (6, split_and_normalize.main),
    (7, train.main),
    (8, evaluate.main),
    (9, export_c.main),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-step", type=int, default=1, help="skip steps before this one (their artifacts must exist)")
    parser.add_argument("--to-step", type=int, default=9)
    args = parser.parse_args()

    for number, step_main in STEPS:
        if args.from_step <= number <= args.to_step:
            print("\n" + "=" * 70 + "\nSTEP {}\n".format(number) + "=" * 70)
            step_main([])


if __name__ == "__main__":
    main()
