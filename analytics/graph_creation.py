
import json, matplotlib.pyplot as plt, numpy as np
with open("cycles.json","r") as f:
    cycles = json.load(f)

durs = [c["dur"] for c in cycles]
if not durs:
    print("No cycles")
else:
    plt.hist(durs, bins=8,density=True)
    plt.title("Cycle time distribution (s)")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.savefig("cycle_hist.png", dpi=150)
    print("Saved cycle_hist.png, median:", np.median(durs))