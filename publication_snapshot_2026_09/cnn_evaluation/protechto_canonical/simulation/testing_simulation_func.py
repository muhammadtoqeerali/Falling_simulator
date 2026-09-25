
import numpy as np


def check_consecutive_ones_with_offset(y_pred, y_true, threshold=2):

    # If both does not contain falls the simulation is passed
    if 1 not in y_true and 1 not in y_pred:
        return True

    # If the session contains a fall, but the model does not predict any fall the simulation is failed
    if 1 in y_true and 1 not in y_pred:
        return False

    # The session doesn't contain falls, but the prediction contains falls, we need to check if we have consecutive ones
    if 1 not in y_true and 1 in y_pred:
        for i in range(y_pred.shape[0]):
            if y_pred[i] == 1:
                if np.sum(y_pred[i: i+threshold]) >= threshold:
                    # Enough fall predictions to trigger a fall, but it's not a fall
                    return False
        return True

    if 1 in y_true and 1 in y_pred:
        for i in range(y_pred.shape[0]):
            if y_pred[i] == 1:
                if np.sum(y_pred[i: i + threshold]) >= threshold and 1 in y_true[i: i+threshold]:
                    # Enough fall predictions to trigger a fall, and a fall is present in the true labels
                    return True

        return False


def test_function():
    tests = {
        "t1":{
            "y_pred": np.array([0, 0, 0, 1, 0, 0]),
            "y_true": np.array([0, 0, 0, 0, 0, 0]),
            "result": True
        },
        "t2": {
            "y_pred": np.array([0, 0, 1, 1, 0, 0]),
            "y_true": np.array([0, 0, 0, 0, 0, 0]),
            "result": False
        },
        "t3": {
            "y_pred": np.array([0, 0, 1, 0, 1, 0]),
            "y_true": np.array([0, 0, 0, 0, 0, 0]),
            "result": True
        },
        "t4": {
            "y_pred": np.array([0, 0, 1, 0, 1, 1]),
            "y_true": np.array([0, 0, 0, 0, 0, 0]),
            "result": False
        },
        "t5": {
            "y_pred": np.array([0, 0, 0, 0, 1, 1]),
            "y_true": np.array([0, 0, 0, 1, 1, 1]),
            "result": True
        },
        "t6": {
            "y_pred": np.array([0, 0, 0, 0, 1, 0]),
            "y_true": np.array([0, 0, 0, 1, 1, 1]),
            "result": False
        },
        "t7": {
            "y_pred": np.array([0, 0, 0, 0, 1, 1]),
            "y_true": np.array([0, 0, 0, 0, 0, 1]),
            "result": True
        },
        "t8": {
            "y_pred": np.array([0, 0, 0, 0, 1, 1]),
            "y_true": np.array([0, 0, 1, 1, 0, 0]),
            "result": False
        }

    }

    for test_name, test_data in tests.items():
        y_pred = test_data["y_pred"]
        y_true = test_data["y_true"]
        is_passed = check_consecutive_ones_with_offset(y_pred, y_true, 2)

        print(f"Test: {test_name}")
        print(f"    y_pred: {[int(i) for i in y_pred]}")
        print(f"    y_true: {[int(i) for i in y_true]}")
        print(f"    Simulation result: {'passed' if is_passed else 'failed'} | Expected: {'passed' if test_data['result'] else 'failed'}")


if __name__ == "__main__":
    test_function()
