import argparse
import logging
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Changes 
# 1. Refactor for Redability, Single Responsiblility and DRY principles
# 2. Vectorise code for optimisation
# 3. Add logging
# 4. Make model parameters user accessible
# 5. Store model weights


def train(input_data, n_iterations, init_learning_rate, width, height, record_frames=True):
    frames = [] if record_frames else None

    neighborhood_radius_init = max(width, height) / 2
    weights = np.random.random((width, height, 3))
    time_const = n_iterations / np.log(neighborhood_radius_init)
    x = np.arange(width)
    y = np.arange(height)
    x_coords, y_coords = np.meshgrid(x, y, indexing='ij')

    for step_idx in range(n_iterations):
        decay_factor = np.exp(-step_idx/time_const)
        neighborhood_radius = neighborhood_radius_init * decay_factor
        learning_rate = init_learning_rate * decay_factor
        for input_vec in input_data:
            weights = update_weight(input_vec, weights, width, height, x_coords, y_coords, neighborhood_radius, learning_rate)
        
        if record_frames and step_idx % max(n_iterations // 100, 1) == 0:
            frames.append(weights.copy())
    return weights, frames

def update_weight(input_vec: np.array, weights: np.array,  width: int, height: int, x_coords: np.array, y_coords: np.array, neighborhood_radius: int, learning_rate: float, ):
    """Update the weight matrix for a given input vector."""
    bmu = np.argmin(np.sum((weights - input_vec) ** 2, axis=2))
    bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
    
    # Caluculate distance between bmu to each nodes
    dist = calculate_distance(bmu_x, bmu_y, x_coords, y_coords)
    # Calculate influence
    influence_factor = np.exp(-(dist ** 2) / (2*(neighborhood_radius ** 2)))
    # Update weights
    weights += learning_rate * influence_factor[:, :, np.newaxis] * (input_vec - weights)
    return weights

def calculate_distance(bmu_x: np.array, bmu_y: np.array, x: np.array, y: np.array):
    """Compute Euclidean distance between each node and the BMU."""
    return np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--height', type=int, default=10)
    parser.add_argument('--init_lr', type=float, default=0.1)
    args = parser.parse_args()
    
    logger.info(f"Training SOM with size {args.width}x{args.height} for {args.iterations} iterations")
    
    # Generate data
    input_data = np.random.random((args.n_samples, args.input_size))

    image_data, frames = train(input_data, args.iterations, args.init_lr, args.width, args.height, args.input_size)
    plt.imsave('1000.png', image_data)

    fig, ax = plt.subplots()
    im = plt.imshow(frames[0], animated=True)
    plt.show()
    ani = FuncAnimation(fig, lambda i: im.set_array(frames[i+1]), interval=200, blit=False, repeat=False, frames=len(frames))
    plt.show()

    gif_path = "./som_weight_update.gif"
    ani.save(gif_path, writer='pillow')
    
if __name__ == '__main__':
    main()
