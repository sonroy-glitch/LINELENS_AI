import json
import matplotlib.pyplot as plt
import numpy as np

with open("changes.json", "r") as f:
    changes = json.load(f)

times = []
labels = []

for item in changes:
    for key, value in item.items():
        labels.append(key)
        times.append(value)

times = np.array(times)
times = times - times[0]

y = [1 if label == "Hazard" else 2 for label in labels]

plt.figure()
plt.plot(times, y, marker='o')

for i in range(len(times)):
    plt.text(times[i], y[i], labels[i])

plt.yticks([1, 2], ["Hazard", "Machine"])
plt.xlabel("Time (seconds from start)")
plt.ylabel("Event Type")
plt.title("Event Timeline")
plt.savefig("changes_plot.png", dpi=150)
plt.show()