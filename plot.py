import matplotlib.pyplot as plt
import json

import argparse
parser = argparse.ArgumentParser(description='Very simple plotting script.')
parser.add_argument('--id', action="store", dest="id")
parser.add_argument('--metric', action="store", dest="metric", default='result.true')

# Should be grouped by label x condition1 x condition2
def plot_metric(id, metric_name):
    metric_file = "./db/" + str(id) + "/metrics.json"
    config_file = "./db/" + str(id) + "/config.json"
    with open(metric_file) as f:
        metrics = json.load(f)

    with open(config_file) as f:
        config = json.load(f)

    x = metrics[metric_name]['steps']
    y = metrics[metric_name]['values']
    plt.plot(x, y)
    plt.xlabel("Frames")
    plt.ylabel(metric_name)
    plt.savefig("Result.png")

    # This is for the ProcGen experiments:
    # Out of the 4 ids, two are emtpy, one is the test performance one is the training performance
    try:
        is_test_worker = config['is_test_worker']
        print("Plotting performance on {} levels".format(
            "test" if is_test_worker else "train"
        ))
    except:
        pass

if __name__ == "__main__":
    args = parser.parse_args()
    plot_metric(args.id, args.metric)