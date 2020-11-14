import json 
import os 


def record_results(best_test):
    # reading in the data from the existing file.
    with open("results.json", "r+") as f:
        data = json.load(f)
        data.append(best_test)
        f.seek(0)
        json.dump(data, f)
    return 


if __name__ == "__main__":
    best_test = {
        "best-epoch": 0,
        "imbalanced": True,
        "learning_rate": 0.001,
        "loss": False,
        "loss_metric": "approx-f1",
        "run_name": "testoutfile",
        "seed": 45,
        "test_accuracy": 0,
        "test_wt_f1_score": 0,
        "val_accuracy": 0,
        "val_wt_f1_score": 0
    }
    record_results(best_test)
