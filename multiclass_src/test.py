import os
import json

def record_results(results_path, best_test, output_file):
    # reading in the data from the existing file.
    file_path = "/".join([results_path, output_file])
    if os.path.isfile(file_path):
        with open(file_path, "r+") as f:
            data = json.load(f)
            data.append(best_test)
            f.close()
        with open(file_path, "w") as outfile:
            json.dump(data, outfile)
    ## if the file doesn't eixst:
    else:
        best_test = [best_test]
        with open(file_path, "w") as outfile:
            json.dump(best_test, outfile)


btest = {"a": 2, "b": 3}
record_results(results_path="/home/tdl29/cs-490-heaviside/multiclass_src/results/new_runs",
        best_test=btest, output_file="test.json")
