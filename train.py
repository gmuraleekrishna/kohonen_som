from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


## Changes 
# 1. rename unicode variables to readable ascii
# 2. move `distance`logic to a method 
# 3. Vectorize `for` loop
# 4. Refactor for readability

def train(input_data, n_max_iterations, width, height):
    neighborhood_radius_init = max(width, height) / 2
    init_learning_rate = 0.01
    weights = np.random.random((width, height, 3))
    time_const = n_max_iterations / np.log(neighborhood_radius_init)
    frames = [weights]
    x = np.arange(width)
    y = np.arange(height)
    x_coords, y_coords = np.meshgrid(x, y, indexing='ij')

    for step_idx in range(n_max_iterations):
        decay_factor = np.exp(-step_idx/time_const)
        neighborhood_radius = neighborhood_radius_init * decay_factor
        learning_rate = init_learning_rate * decay_factor
        for vt in input_data:
            weights = update_weight(width, height, weights, x_coords, y_coords, neighborhood_radius, learning_rate, vt)
        frames.append(weights)
    return weights, frames

def update_weight(width, height, weights, x_coords, y_coords, neighborhood_radius, learning_rate, vt):
    bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
    bmu_x, bmu_y = np.unravel_index(bmu, (width, height))

    dist = calculate_distance(bmu_x, bmu_y, x_coords, y_coords)
    influence_factor = np.exp(-(dist ** 2) / (2*(neighborhood_radius ** 2)))
    weights += learning_rate * influence_factor[:, :, np.newaxis] * (vt - weights)
    return weights

def calculate_distance(bmu_x, bmu_y, x, y):
    return np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))


if __name__ == '__main__':
    import timeit
    # Generate data
    input_data = np.random.random((5,3))
    start_time = timeit.default_timer()
    image_data, frames = train(input_data, 100, 10, 10)
    end_time = timeit.default_timer()
    print(f"Duration 1: {end_time - start_time}s")
    start_time = end_time
    plt.imsave('100.png', image_data)

    # Generate data
    image_data, frames = train(input_data, 10000, 100, 100)
    end_time = timeit.default_timer()
    print(f"Duration 2: {end_time - start_time}s")

    fig, ax = plt.subplots()
    im = plt.imshow(frames[0], animated=True)
    plt.show()

    ani = FuncAnimation(fig, lambda i: im.set_array(frames[i+1]), interval=200, blit=False, repeat=False, frames=len(frames))
    plt.show()

    gif_path = "./som_weight_update.gif"
    ani.save(gif_path, writer='pillow')
    plt.imsave('1000.png', image_data)