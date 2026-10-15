"""Generate the final hard-voting submission without rerunning CV."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import SUBMISSION_PATH, TARGET_COLUMN
from main import create_submission, load_data


def main() -> None:
    """Fit the final ensemble on all training data and save predictions."""
    train_data, test_data = load_data()
    submission = create_submission(train_data, test_data)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved to: {SUBMISSION_PATH}")
    print(submission[TARGET_COLUMN].value_counts().sort_index())


if __name__ == "__main__":
    main()
