import numpy as np
data = np.load("../data/ns_unsteady.npy", allow_pickle=True).item()
for key in data.keys():
    print(f"{key}: {type(data[key])}, shape: {np.shape(data[key])}")

