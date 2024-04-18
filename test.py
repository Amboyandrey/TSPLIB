import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class TSPSolver:
    def __init__(self, points):
        self.points = np.array(points)
        self.n = len(points)
        self.tour = None

    def nearest_neighbor(self, start=None):
        if start is None:
            start = np.random.randint(0, self.n)
        unvisited = set(range(self.n))
        unvisited.remove(start)
        tour = [start]

        while unvisited:
            current_city = tour[-1]
            nearest_city = min(unvisited, key=lambda city: np.linalg.norm(self.points[current_city] - self.points[city]))
            tour.append(nearest_city)
            unvisited.remove(nearest_city)

        self.tour = tour

    def total_distance(self, tour=None):
        if tour is None:
            tour = self.tour
        return sum(np.linalg.norm(self.points[tour[i]] - self.points[tour[i + 1]]) for i in range(self.n - 1))

    def two_opt_swap(self, tour, i, k):
        new_tour = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
        return new_tour

    def two_opt(self, tour=None):
        if tour is None:
            tour = self.tour

        improvement = True
        iteration = 0
        while improvement:
            improvement = False
            for i in range(1, self.n - 2):
                for k in range(i + 1, self.n):
                    iteration += 1
                    if k - i == 1:
                        continue  # Changes nothing, skip
                    new_tour = self.two_opt_swap(tour, i, k)
                    if self.total_distance(new_tour) < self.total_distance(tour):
                        tour = new_tour
                        improvement = True
                        yield tour, self.total_distance(tour), iteration  # Yield the improved tour, current distance, and iteration for visualization

        self.tour = tour

    def solve(self):
        self.nearest_neighbor()
        for improved_tour, current_distance, iteration in self.two_opt():
            # Yield each improved tour, current distance, and iteration
            yield improved_tour, current_distance, iteration
        
        # Connect the last city back to the starting city to form a closed tour
        self.tour.append(self.points[self.tour[0]])

# Parse the TSP dataset and extract coordinates
def parse_tsp_dataset(data):
    coordinates = []
    pattern = re.compile(r'(\d+)\s+([\d.]+)\s+([\d.]+)')

    in_node_coord_section = False
    for line in data.split('\n'):
        if line.startswith("NODE_COORD_SECTION"):
            in_node_coord_section = True
            continue
        if line.startswith("EOF"):
            break
        if in_node_coord_section:
            match = pattern.match(line)
            if match:
                coordinates.append((float(match.group(2)), float(match.group(3))))

    return coordinates

# Load the TSP dataset from a file
def load_tsp_file(file_path):
    with open(file_path, 'r') as file:
        tsp_data = file.read()
    return tsp_data

# Specify the path to your TSP file
tsp_file_path = "/home/andreyaa/Desktop/TSPLIB/TSPLIB/EUC_2D/ch130.tsp"

# Load the TSP dataset from the file
tsp_data = load_tsp_file(tsp_file_path)

# Parse the dataset and extract coordinates
city_coordinates = parse_tsp_dataset(tsp_data)

# Store total distances
total_distances = []
render_times = []

# Create TSPSolver instance
solver = TSPSolver(city_coordinates)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(solver.points[:, 0], solver.points[:, 1], c='blue', label='Cities')
line, = ax.plot([], [], 'r-', label='Current Tour')
iteration_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
distance_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
plt.title('TSP Solution using Nearest Neighbor with 2-opt')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='upper right')
plt.grid()

def init():
    line.set_data([], [])
    iteration_text.set_text('')
    distance_text.set_text('')
    return line, iteration_text, distance_text

def update(data):
    tour, current_distance, iteration = data
    tour_points = np.array([solver.points[i] for i in tour])
    
    # Connect the first and last points to close the loop
    tour_points = np.vstack((tour_points, tour_points[0]))
    
    line.set_data(tour_points[:, 0], tour_points[:, 1])
    iteration_text.set_text(f'Current Iteration: {iteration}')
    distance_text.set_text(f'Current Distance: {current_distance:.2f}')
    return line, iteration_text, distance_text


# Animate the improvements in the tour
ani = animation.FuncAnimation(fig, update, frames=solver.solve(), init_func=init, blit=True)

plt.show()
