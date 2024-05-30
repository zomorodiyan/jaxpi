import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("ns_unsteady.npy", allow_pickle=True).item()
#data = np.load("fine_mesh.npy", allow_pickle=True).item()
#data = np.load("fine_mesh_near_cylinder.npy", allow_pickle=True).item()
for key in data.keys():
    print(f"{key}: {type(data[key])}, shape: {np.shape(data[key])}")
coords = data["coords"]

# Extract x and y coordinates
x_coords = coords[:, 0]
y_coords = coords[:, 1]

# Plot the 2D geometry
plt.figure(figsize=(10, 6))
plt.scatter(x_coords, y_coords, s=1, color='blue')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Geometry Visualization')
plt.axis('equal')  # Ensure equal scaling
plt.grid(True)
plt.show()

